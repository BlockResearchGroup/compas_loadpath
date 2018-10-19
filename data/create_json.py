
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compas_tna.diagrams import FormDiagram
from compas.utilities import geometric_key

import rhinoscriptsyntax as rs


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


# Form

guids = rs.ObjectsByLayer('Lines') + rs.ObjectsByLayer('Symmetry')
lines = [[rs.CurveStartPoint(i), rs.CurveEndPoint(i)] for i in guids]
form = FormDiagram.from_lines(lines, delete_boundary_face=False)

form.update_default_vertex_attributes({'is_roller': False})
form.update_default_edge_attributes({'q': 1, 'is_symmetry': False})
form.attributes['loadpath'] = 0
form.attributes['indset'] = []

gkey_key = form.gkey_key()


# Pins

for i in rs.ObjectsByLayer('Pins'):
    gkey = geometric_key(rs.PointCoordinates(i))
    form.set_vertex_attribute(gkey_key[gkey], 'is_fixed', True)
    
# Rollers

for i in rs.ObjectsByLayer('Rollers'):
    gkey = geometric_key(rs.PointCoordinates(i))
    form.set_vertex_attribute(gkey_key[gkey], 'is_roller', True)

# Loads

loads = FormDiagram.from_lines(lines)
for key in loads.vertices():
    form.vertex[key]['pz'] = loads.vertex_area(key=key)

# Constraints

#for i in rs.ObjectsByLayer('Lower'):
#    gkey = geometric_key(rs.PointCoordinates(i))
#    form.set_vertex_attributes(gkey_key[gkey], {'lb': float(rs.ObjectName(i))})
    
#for i in rs.ObjectsByLayer('Upper'):
#    gkey = geometric_key(rs.PointCoordinates(i))
#    form.set_vertex_attributes(gkey_key[gkey], {'ub': float(rs.ObjectName(i))})
    
# Symmetry

for i in rs.ObjectsByLayer('Symmetry'):
    if rs.IsCurve(i):
        u = gkey_key[geometric_key(rs.CurveStartPoint(i))]
        v = gkey_key[geometric_key(rs.CurveEndPoint(i))]
        form.set_edge_attribute((u, v), name='is_symmetry', value=True)

# TextDots

rs.EnableRedraw(False)
rs.DeleteObjects(rs.ObjectsByLayer('Dots'))
rs.CurrentLayer('Dots')

pzt = 0
for key in form.vertices():
    pz = form.vertex[key].get('pz', 0)
    pzt += pz
    if pz:
        rs.AddTextDot('{0:.2f}'.format(pz), form.vertex_coordinates(key))
print('Total load: {0}'.format(pzt))

rs.EnableRedraw(True)
rs.CurrentLayer('Default')

# Save

form.to_json('F:/compas_loadpath/data/arches_curved.json')
