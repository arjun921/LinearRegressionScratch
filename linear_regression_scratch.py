from numpy import *

def compute_error_for_line_given_points(b,m,points):
	totalError = 0
	#for every point
	for i in range(0,len(points)):
		#get the x value
		x = points[i,0]
		#get the y value
		y = points[i,1]
		#get the difference, square it, add it to the total
		totalError+=(y-(m*x+b))**2
	return totalError/float(len(points))

def step_gradient(b_current, m_current,points, learning_rate):
	#starting points for gradients
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):  
		x = points[i,0]
		y = points[i,1]
		#direction with respect to b and m
		#computing partial derivatives of our error functions

		b_gradient += -(2/N)*(y-((m_current*x)+b_current))
		m_gradient += -(2/N)*x*(y-((m_current*x)+b_current))

	new_b = b_current-(learning_rate*b_gradient)
	new_m = m_current-(learning_rate*m_gradient)
	return [new_b,new_m]

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
	#starting b and m
	b = starting_b
	m = starting_m

	#gradient descent
	for i in range(num_iterations):
		#update b and m with more accurate b and m by performing gradient step
		b,m = step_gradient(b,m,array(points), learning_rate)
	return [b,m]



def run():
	#step 1 Collect our data
	#2 loops in genfromtxt
	# each line in file to sequence of strings
	# second loop converting each string to its appropriate data type
	points = genfromtxt('data.csv', delimiter=',')

	#Step 2. Define Hyperparameteres
	# Hyperparameters nothing but a fancy word for parameters
	#how fast should our model converge
	learning_rate = 0.0001
	#y = mx+b(slope formula)
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	# Step 3. Training the model
	print ('starting gradient at b = {}, m={}, error = {}'.format(initial_b,initial_m,compute_error_for_line_given_points(initial_b,initial_m,points)))
	print ('Running...')
	[b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)
	print ('Ending point at {} iterations, b = {}, m={}, error = {}'.format(num_iterations,b,m,compute_error_for_line_given_points(b,m,points))) 

if __name__ == "__main__":
	run()
