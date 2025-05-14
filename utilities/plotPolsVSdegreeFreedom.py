import matplotlib.pyplot as plt
import numpy as np


dataUn={"hive":42,"standard":68,"2CC_hive":114,"2CC_standard":243,"1CC":385}

dataMaxMolt={"hive":402,"standard":290,"2CC_hive":285,"2CC_standard":144,"1CC":116}
#below I'm summing all multiplicity = 501-unique

dataMolt={"hive":459,"standard":433,"2CC_hive":386,"2CC_standard":258,"1CC":116}

n=[0,1,2,3,4]

plt.figure()
plt.ion()

plt.plot(n,dataUn.values(),'o--',lw=2,label="uniquePol",color="tab:blue")

plt.plot(n,dataMolt.values(),'o--',lw=2,label="AllMolt",color="tab:orange")

plt.plot(n,dataMaxMolt.values(),'*--',lw=2,ms=8,label="maxMolt",color="tab:red")

plt.xticks(n,dataUn.keys())
plt.legend()

plt.show()

input()