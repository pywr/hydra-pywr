from pywr.core import Model


if __name__ == '__main__':
    import sys

    m = Model.load(sys.argv[1])
    m.run()

    for node in m.nodes:
        print(node, node.flow)

