"""
Fit a exponential-truncated normal mixture model to a list of peak height T-values.
The model is introduced in the HBM poster: 
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""

class neuropower():
	"""needs to be defined"""

def nulcumdens(exc,peaks):
	"""Returns cumulative  density (p-values) using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005"""
	v = exc
	u = peaks - v
	F0 = 1-((u+v)**2*exp(-(u+v)**2/2)/(v**2*exp(-v**2/2)))
	return F0

def nulprobdens(exc,peaks):
	"""Returns probability density using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005"""
	v = exc
	u = peaks - v
	f0 = (2+(u+v)**2)*(u+v)*exp(-(u+v)**2/2)/(v**2*exp(-v**2/2))
	return f0

def altprobdens(mu,sigma,exc,peaks):
	"""Returns probability density using a truncated normal distribution that we define as the distribution of local maxima in a GRF under the alternative hypothesis of activation"""
	num = 1/sigma*scipy.stats.norm(mu,sigma).pdf(peaks)
	den = 1-scipy.stats.norm(mu,sigma).cdf(exc)
	fa = num/den
	return fa

def mixprobdens(mu,sigma,pi1,exc,peaks):
	"""Returns probability density using mixture of an exponential function and a truncated normal distribution, weighted by pi1 (=1-pi0)"""
	f0 = nulprobdens(exc,peaks)
	fa = altprobdens(mu,sigma,exc,peaks)
	f = (1-pi1)*f0 + pi1*fa
	return(f)

def mixprobdensSLL(pars,pi1,exc,peaks):
	mu=pars[0]
	sigma=pars[1]
	f = mixprobdens(mu,sigma,pi1,exc,peaks)
	LL = -sum(np.log(f))
	return(LL)

def npowerOptim(peaks,pi1,exc):
	"""Searches the maximum likelihood estimator for the mixture distribution of null and alternative
	
	Example
	-------
	Something is still not right with the optimiser
	"""
	
	start = (5,0.5)
	opt = scipy.optimize.minimize(mixprobdensSLL,[5,0.5],method='L-BFGS-B',args=((pi1,exc,[x])),jac=fpLL,bounds=((2.5,50),(0.1,50)))
