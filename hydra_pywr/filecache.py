import os
import s3fs
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

import logging
log = logging.getLogger(__name__)


class CacheEntry():
    def __init__(self, url):
        self.url = None
        self.dest = None

    @property
    def filetype(self):
        pass

    @property
    def is_local(self):
        return os.path.exists(self.dest)



class S3CacheEntry(CacheEntry):
    def __init__(self, url):
        self.url = url
        self.fs = s3fs.S3FileSystem(anon=True)
        self.info = fs.info(self.url)

    def fetch(self, dest):
        log.info(f"Retrieving {self.url} to {dest} ...")
        self.fs.get(self.url, dest)
        log.info(f"Retrieved {dest} ({os.stat(dest).st_size} bytes)")
        self.dest = dest

    @property
    def checksum(self):
        return self.info["ETag"]


class HttpCacheEntry(CacheEntry):
    def __init__(self, url):
        pass


class FsCacheEntry(CacheEntry):
    def __init__(self, url):
        pass


class FileCache():

    scheme_map = {
        "s3": S3CacheEntry,
        "http": HttpCacheEntry,
        "https": HttpCacheEntry,
        "": FsCacheEntry
    }

    def __init__(self, cache_root):
        self.cache_root = cache_root
        self.buf = dict()

    def __len__(self):
        return len(self.buf)

    def __contains__(self, key):
        return key in self.buf

    def add(self, url):
        entry = self.__class__.lookup_type(url)(url)
        dest = self.url_to_local_path(url)
        print(f"{dest=}")
        entry.fetch(dest)
        self.buf[url] = dest
        return entry

    @classmethod
    def lookup_type(cls, path):
        u = urlparse(path)
        scheme_type = cls.scheme_map.get(u.scheme)
        if not scheme_type:
            raise ValueError(f"[{cls.__qualname__}] Unsupported scheme: '{u.scheme}'")

        return scheme_type

    def url_to_local_path(self, url):
        u = urlparse(url)
        filepath = f"{u.netloc}{u.path}"
        return os.path.join(self.cache_root, filepath)

    @property
    def is_current(self):
        pass

    def retrieve_url(self, url):
        """
        if not os.path.exists(urldir):
            try:
                os.makedirs(urldir)
            except OSError as err:
                raise OSError(f"Unable to create URL retrieval directory at {urldir}: {err}")
        elif not os.path.isdir(urldir):
            raise OSError(f"Destination '{urldir}' is not a directory")
        """

        filedest = self.url_to_local_path(url)
        log.info(f"Retrieving {url} to {filedest} ...")

        with urlopen(url) as resp, open(filedest, "wb") as fp:
            shutil.copyfileobj(resp, fp)

        log.info(f"Retrieved {filedest} ({os.stat(filedest).st_size} bytes)")
        return filedest

    def retrieve_s3(s3path):
        filedest = self.url_to_local_path(s3path)

        """
        if not os.path.exists(datadir):
            try:
                os.makedirs(datadir)
            except OSError as err:
                raise OSError(f"Unable to create S3 retrieval directory at {datadir}: {err}")
        elif not os.path.isdir(datadir):
            raise OSError(f"Destination '{datadir}' is not a directory")
        """

        fs = s3fs.S3FileSystem(anon=True)
        log.info(f"Retrieving {s3path} to {filedest} ...")
        fs.get(s3path, filedest)
        log.info(f"Retrieved {filedest} ({os.stat(filedest).st_size} bytes)")
        return filedest


if __name__ == "__main__":
    s3_url = "s3://modelers-data-bucket/eapp/single/ETH_flow_sim.h5"
    fc = FileCache("/tmp/cache")
    print(f"{s3_url in fc=}")
    s3_cached = fc.add(s3_url)
    print(s3_cached.is_local)
