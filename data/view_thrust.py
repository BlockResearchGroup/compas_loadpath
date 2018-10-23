
from compas_tna.diagrams import FormDiagram
from compas_rhino.artists import MeshArtist


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


form = FormDiagram.from_json('F:/temp/output.json')
print(form.attributes['loadpath'])

artist = MeshArtist(form, layer='Thrust')
artist.clear_layer()
artist.draw_edges()
