
import os
from random import randbytes
from . import ResultsProcessor
import logging
import datetime
import uuid

import os
log = logging.getLogger(__name__)

class MongoResultsProcessor(ResultsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initialize the MongoDB results processor
        Use the following env variables 
        export USE_MONGO=true
        export MONGO_URI="<URI>"
        export MONGO_USERNAME="<USERNAME>"
        export MONGO_PASSWORD="<PASSWORD>"
        export MONGO_DATABASE="hydra"
        export MONGO_MODEL_RESULTS_COLLECTION="model_results"
        Find credentials here: https://gitlab.hydra.org.uk/hwi/docs/-/wikis/MongoDB-Docker-Service
        """
        self.mongo_database = os.getenv("MONGO_DATABASE", "hydra_data")
        self.mongo_collection = os.getenv("MONGO_MODEL_RESULTS_COLLECTION", "model_results")
        self.mongo_client = None
        # Generate unique run ID for this model run
        self.run_id = str(uuid.uuid4())
        self.run_timestamp = datetime.datetime.now()
        # List to store all documents for the model_results collection
        self.model_results_documents = []
        #not to be confused with hydra attributes, these are attribute definitions to be stored in the mongo collection
        self.mongo_attribute_definition_dict = {}

        self.resource_results_dict = {}

    def connect(self):
        """
            Check if mongo credentials are available in the ENV variables. If so,
            instantiate a client, and attempt to connect. If connection is successful
            return the mongo client
        """
        try:
            import pymongo
        except ImportError:
            log.error("pymongo is not installed")
            return None
    
        mongo_uri = os.getenv("MONGO_URI")
        mongo_username = os.getenv("MONGO_USERNAME")
        mongo_password = os.getenv("MONGO_PASSWORD")
        use_mongo = os.getenv("USE_MONGO", "true").lower() in ("1", "true", "yes")

        if not use_mongo:
            log.info("MongoDB is disabled via USE_MONGO environment variable.")
            return None

        if mongo_uri:
            self.mongo_client = pymongo.MongoClient(mongo_uri,
                                         username=mongo_username,
                                         password=mongo_password)

            try:
                self.mongo_client.admin.command('ping')
                log.info("Connected to MongoDB..checking for database")

                log.info("Connected to MongoDB database: %s", self.mongo_database)

                return self.mongo_client
            except Exception as e:
                log.error("Error connecting to MongoDB: %s", e)

        return None

    def create_indexes(self):
        """
        Create the recommended indexes for efficient querying
        Only creates indexes if they don't already exist
        """
        if self.mongo_client is None:
            log.error("MongoDB client is not connected")
            return

        collection = self.mongo_client[self.mongo_database][self.mongo_collection]
        
        # Get existing index information
        existing_indexes = collection.index_information()
        existing_index_names = set(existing_indexes.keys())
        
        indexes_to_create = []
        
        # Define all indexes we want to create
        desired_indexes = [
            {
                "keys": [("scenario_id", 1), ("run_timestamp", -1), ("element_type", 1)],
                "name": "scenario_latest_type",
                "description": "Primary index for most recent run queries"
            },
            {
                "keys": [("scenario_id", 1), ("run_id", 1), ("element_type", 1)],
                "name": "scenario_run_type", 
                "description": "Index for historical run lookups"
            },
            {
                "keys": [("scenario_id", 1), ("run_timestamp", -1), ("element_id", 1)],
                "name": "scenario_latest_element",
                "description": "Index for specific element queries"
            },
            {
                "keys": [("run_id", 1)],
                "name": "run_data",
                "description": "Index for full run data retrieval"
            }
        ]
        
        # Check which indexes need to be created
        for index_spec in desired_indexes:
            if index_spec["name"] not in existing_index_names:
                indexes_to_create.append(index_spec)
                log.info("Will create index: %s - %s", index_spec["name"], index_spec["description"])
            else:
                log.info("Index already exists: %s", index_spec["name"])
        
        # Create missing indexes
        created_count = 0
        for index_spec in indexes_to_create:
            try:
                collection.create_index(index_spec["keys"], name=index_spec["name"])
                created_count += 1
                log.info("Created index: %s", index_spec["name"])
            except Exception as e:
                log.error("Failed to create index %s: %s", index_spec["name"], e)
        
        if created_count > 0:
            log.info("Created %d new indexes for %s collection", created_count, self.mongo_collection)
        else:
            log.info("All required indexes already exist for %s collection", self.mongo_collection)

    def get_latest_run_results(self, scenario_id, element_type=None):
        """
        Get results from the most recent run for a given scenario
        
        Args:
            scenario_id: The scenario ID to query
            element_type: Optional filter by element type (node, link, network, etc.)
            
        Returns:
            List of documents from the most recent run
        """
        if self.mongo_client is None:
            log.error("MongoDB client is not connected")
            return []

        collection = self.mongo_client[self.mongo_database][self.mongo_collection]
        
        # Build query
        query = {"scenario_id": scenario_id}
        if element_type:
            query["element_type"] = element_type

        # Get the most recent run_id for this scenario
        latest_run = collection.find_one(
            {"scenario_id": scenario_id},
            {"run_id": 1, "run_timestamp": 1},
            sort=[("run_timestamp", -1)]
        )
        
        if not latest_run:
            log.warning("No runs found for scenario %s", scenario_id)
            return []

        # Get all results for the latest run
        query["run_id"] = latest_run["run_id"]
        results = list(collection.find(query))
        
        log.info("Retrieved %d documents from latest run %s for scenario %s", 
                len(results), latest_run["run_id"], scenario_id)
        
        return results

    def get_run_history(self, scenario_id, limit=10):
        """
        Get historical run information for a scenario
        
        Args:
            scenario_id: The scenario ID to query
            limit: Maximum number of runs to return
            
        Returns:
            List of run metadata documents
        """
        if self.mongo_client is None:
            log.error("MongoDB client is not connected")
            return []

        collection = self.mongo_client[self.mongo_database][self.mongo_collection]
        
        # Get unique runs for this scenario, sorted by most recent
        pipeline = [
            {"$match": {"scenario_id": scenario_id, "element_type": "scenario"}},
            {"$sort": {"run_timestamp": -1}},
            {"$limit": limit}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        log.info("Retrieved %d historical runs for scenario %s", len(results), scenario_id)
        
        return results

    def save(self):
        """
        Save the results to MongoDB using the model_results collection schema
        """
        if self.mongo_client is None:
            log.error("MongoDB client is not connected")
            return

        # Create scenario document
        scenario_doc = self.make_scenario_document()
        self.model_results_documents.append(scenario_doc)
        
        # Create bucket document
        bucket_doc = self.make_bucket_document()
        self.model_results_documents.append(bucket_doc)

        # Process results and create node/link/network documents
        self.process_results()

        # Add attribute definition documents
        for attr_def in self.mongo_attribute_definition_dict.values():
            attr_doc = self.make_attribute_document(attr_def)
            self.model_results_documents.append(attr_doc)

        # Add resource (node/link/network) documents
        for resource_data in self.resource_results_dict.values():
            resource_doc = self.make_resource_document(resource_data)
            self.model_results_documents.append(resource_doc)

        # Insert all documents into the model_results collection
        if self.model_results_documents:
            collection = self.mongo_client[self.mongo_database][self.mongo_collection]
            result = collection.insert_many(self.model_results_documents)
            log.info("Inserted %d documents for run %s into %s collection", 
                    len(result.inserted_ids), self.run_id, self.mongo_collection)
        else:
            log.warning("No documents to save for run %s", self.run_id)

    def make_scenario_document(self):
        """
        Create a scenario document for the model_results collection
        """
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "element_type": "scenario",
            "element_id": str(self.scenario_id),
            "cr_date": self.run_timestamp
        }

    def make_bucket_document(self):
        """
        Create a bucket document for the model_results collection
        """
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "element_type": "bucket",
            "element_id": self.bucket_name,
            "prefix": self.s3_path
        }

    def make_attribute_document(self, attr_def):
        """
        Create an attribute definition document for the model_results collection
        """
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "element_type": "attribute",
            "element_id": attr_def["_id"],
            "label": attr_def.get("label"),
            "description": attr_def.get("description"),
            "unit": attr_def.get("unit")
        }

    def make_resource_document(self, resource_data):
        """
        Create a resource (node/link/network) document for the model_results collection
        """
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "element_type": resource_data["type"],
            "element_id": str(resource_data["_id"]),
            "name": resource_data.get("name"),
            "attributes": resource_data.get("attributes", {})
        }

    def make_attribute_definition_dict(self, id, label=None, description=None, unit=None):
        """
        Make an attribute definition dict for internal processing
        """
        #if label is None, turn it into a human-readable string
        if label is None:
            label = id.replace("_", " ").title()

        return {
            "_id": id,
            "type": "attribute",
            "label": label,
            "description": description,
            "unit": unit,
            "cr_date": self.run_timestamp
        }

    def make_bucket_dict(self):
        """
        Make a bucket dict for internal processing (kept for compatibility)
        """
        return {
            "_id": self.bucket_name,
            "type": "bucket",
            "prefix": self.s3_path
        }


    def process_results(self):

        for recorder in self.df_recorders:
            recorder_data = self.process_df_recorder(recorder)
            if recorder_data is None:
                continue
            try:
                hydra_node = self._get_hydra_node_from_recorder(recorder)
            except AttributeError:
                hydra_node = None

            resourcetype = 'node' if hydra_node else 'network'
            resourceid = hydra_node.id if hydra_node else self.hydra_network.id
            #make a key from the recorder name, like 'Simulated Flow' -> 'simulated_flow'
            if ':' not in recorder.name:
                recorder_key = recorder.name.replace(" ", "_").replace("-", "_").lower()
            else:
                recorder_key = recorder.name.split(':')[1].replace(" ", "_").replace("-", "_").lower()
            attribute_dict = self.make_attribute_dict(recorder_key, recorder_data)

            if self.mongo_attribute_definition_dict.get(recorder_key) is None:
                self.mongo_attribute_definition_dict[recorder_key] = self.make_attribute_definition_dict(
                    recorder_key,
                    label=None, # Auto-generate this
                    description=recorder_data.get("description"),
                    unit=recorder_data.get("unit")
                )

            if (resourceid, resourcetype) not in self.resource_results_dict:
                self.resource_results_dict[(resourceid, resourcetype)] = {
                    "type": resourcetype,
                    "_id": resourceid,
                    "name": hydra_node.name if hydra_node else self.hydra_network.name,
                    "attributes": {recorder_key: attribute_dict}
                }
            else:
                self.resource_results_dict[(resourceid, resourcetype)]["attributes"][recorder_key] = attribute_dict

        return self.resource_results_dict

    def make_attribute_dict(self, id, recorder_data):
        """
        Make a document that looks like:
        {
            "value": "{buckets:bucket1}/{id}.h5",
            "metadata": {
                "xAxisLabel": "{id.replace('_', ' ').title()}"
            }
        }
        """
        return {
            "value": recorder_data["data"],
            "metadata": recorder_data.get("metadata", {}),
        }