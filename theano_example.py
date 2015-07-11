import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

w = T.dmatrix('w')
v = T.dmatrix('v')
u = w + v
g = function([w, v], u)

print f(2, 3)
print g([[1,3],[2,4]], [[2,4],[1,3]])

a = T.vector()
b = T.vector()
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = function([a, b], out)  # compile function
print f([0, 1, 2], [2, 1, 0])  # prints `array([0, 2, 1026])`