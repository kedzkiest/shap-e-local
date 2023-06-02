from vedo import *
import sys

mesh = Mesh(sys.argv[1] + ".obj")

mesh.show()