from fxpmath import Fxp

# x = Fxp(-7.25391, True, 12, 8)
# x.info(verbose=3)
# print(x)

# print(x.bin(frac_dot=True))
# print(x.base_repr(2))
# print(x.bin())


# x1 = Fxp(-7.25, signed=True, n_word=16, n_frac=8)
# x2 = Fxp(1.5, signed=True, n_word=16, n_frac=8)
# x3 = Fxp(-0.5, signed=True, n_word=8, n_frac=7)


# y = Fxp(None, signed=True, n_word=32, n_frac=16)
# y( (x1*x3 - 3*x2)() )     

# y.info(verbose=3)

x1 = Fxp(0.5, signed=True, n_word=3, n_frac=1)
x2 = Fxp(1.5, signed=True, n_word=3, n_frac=1)
# x1.info(verbose=3)
# x2.info(verbose=3)

y1 = Fxp(None, signed=True, n_word=3, n_frac=1)
# y1.info(verbose=3)
y1.equal(x1 + x2)

y1.info(verbose=3)

print(y1.get_status())
print(type(y1.get_status()))
print(y1.get_status(format=str))
print(type(y1.get_status(format=str)))

s = y1.get_status(format=str)



