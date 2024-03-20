from math import exp, sqrt, sin, cos, log

""" Define the function that we'll find the roots of
"""
def f(x):
	return exp(x-sqrt(x)) - x

""" Implementation of Newton's method.
      (1) f: function that we want the root of
      (2) x_start: initial guess for root
      (3) max_itrs (optional): maximum number of times to update x_n
      (4) tol (optional): how far from the root are we willing to tolerate
    Output:
      (1) x_n: last guess of location of root
"""
def newton(f,x_start,max_itrs=200,tol=1e-8):

	x_old = x_start + 2*max(tol,x_start)
	x_n = x_start

	# Main Newton's method loop
	for i in range(1, max_itrs):
		# Update our function value for the new x location
		f_x = f(x_n)

		# If our x is barely changing, we've found the root
		if abs(x_n-x_old) < tol:
			return x_n
		# Otherwise, update our guess
		else:
			fprime = (f(x_n*1.0001) - f(x_n*0.9999))/(0.0002*x_n)
			x_old = x_n
			x_n -= f_x/fprime

		# Print an update statement for this iteration
		print(i,x_n,f_x)

	print("Exceeded iteration limit without solution")
	return None


""" By putting our main function inside this if statement, we can safely
      import the module from other scripts without having this code execute
      every time
"""
if __name__ == '__main__':
	root = newton(f,1.5)