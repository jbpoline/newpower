"""
Fit a beta-uniform mixture model to a list of p-values.
The BUM model is introduced in Pounds & Morris, 2003.
"""

class BUM():
		
	""" Introduces the class containing the output of the BUM model
	
	Parameters
	----------
	value : float64
		The value of the negative sum of the loglikelihood
	a : float64
		The first shape parameter defining the shape of the beta-distribution
	l : float64
		The second shape parameter defining the mixing of the uniform and the beta-distribution
	pi0 : float64
		Using the shape parameters, we can compute the proportion of the pvalues coming from the uniform distribution
	"""
		
	def __init__(self,value,a,l,pi0):
		self.value = value
		self.a = a
		self.l = l
		self.pi0 = pi0

def fpLL(pars,x):
	"""Returns the gradient function of the BUM model"""
	a = pars[0]
	l = pars[1]
	dl = -sum((1-a*x**(a-1))/(a*(1-l)*x**(a-1)+l))
	da = -sum((a*(1-l)*x**(a-1)*np.log(x)+(1-l)*x**(a-1))/(a*(1-l)*x**(a-1)+l))
	return np.asarray([dl,da])

def fbum(pars,x):
	""" Returns the likelihood of each p-value with the beta uniform mixture model with shape parameters in pars"""
	a = pars[0]
	l = pars[1]
	ret = l+(1-l)*a*x**(a-1)
	return ret	

def fbumnLL(pars,x):
	"""Returns the negative sum of the loglikelihood"""
	return(-fbumLL(pars,x))

def fbumLL(pars,x):
	""" Returns the sum of the loglikelihood"""
	sumlog = sum(np.log(fbum(pars,x)))
	return(sumlog)

def bumOptim(x,starts=1):
	"""Searches the maximum likelihood estimator for the shape parameters of the BUM-model given a list of p-values
	
	Example
	-------
	p = np.random.uniform(0.00001,0.1,100)
	p = np.append(p,np.random.uniform(0.00001,1,100))
	BUM=bumOptim(p,starts=100)
	BUM.pi0
	"""
	
	a = np.random.uniform(0.05,0.95,(starts,))
	l = np.random.uniform(0.05,0.95,(starts,))
	best = []
	par = []
	x = np.asarray(x)
	for i in range(0,starts):
		pars = np.array((a[i],l[i]))
		opt = scipy.optimize.minimize(fbumnLL,[pars[0],pars[1]],method='L-BFGS-B',args=[y],jac=fpLL,bounds=((0.00001,3),(0.00001,3)))
		best.append(opt.fun)
		par.append(opt.x)
	minind=best.index(np.nanmin(best))
	bestpar=par[minind]
	pi0=bestpar[1] + (1-bestpar[1])*bestpar[0]
	out = BUM(best[minind],bestpar[0],bestpar[1],pi0)
	return(out)
