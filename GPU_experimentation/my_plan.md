PART 1 - Pythtb building

GPU speedup for PythTB WIlson Loop calculator(Newton-Shultz)
Basic benchmarks completed
Building the Kane-Mele Setup 
Calculating wilson loop with basic settup?
Building the Kane-Mele Setup with Anderson Disorder
Can we again figure out the topological insulation properties in some way at this step(verifying understanding)?
Parallelizing to simulate many setups at once with CPU- Monte-Carlo simulation
Parallelizing to simulate many setups at once with GPU(second layer of GPU improvements) - Monte-Carlo simulation
Benchmark the difference (between unoptimized WIlson Loop calculator and also unoptimized batching, just simple loop) - this should give a hefty speed up number
	PART2 - Calculating the topological phase transition
Simulate several system sizes over a range of disorder coefficients 
Use Finite size scaling (Binder Parameter?) to get an estimate for critical value?
Compare against values derived from paper todo: figure out what this exact benchmark is. Make it well defined. We are using a different method so there is a high chance of it being off.
This creates an argument for GPU parallelization being useful
We should also mention/ compare against BerryEasy or spinV if we have loads of extra credits. But would be really good.
