import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

critic_loss_data = np.loadtxt('./model/simple/critic_loss.txt', dtype=np.float32, delimiter=',')
data1 = pd.read_table('./model/simple/critic_loss.txt',header=None, delim_whitespace=True)
data11 = data1.values
df_critic_loss = pd.DataFrame(data11)
# print("critic_loss_data的长度")
# print(len(critic_loss_data))
# print("df_critic_loss的长度")
# print(len(df_critic_loss))

plt.figure("critic loss")
plt.plot(range(len(critic_loss_data)), df_critic_loss[0].rolling(50, min_periods=1).mean())
plt.xlabel('time_step ')
plt.ylabel('critic_loss smooth')
plt.savefig('./model/simple' + '/critic_loss_smooth.pdf',  bbox_inches="tight", format='pdf')

actor_loss_data = np.loadtxt('./model/simple/actor_loss.txt', dtype=np.float32, delimiter=',')
data2 = pd.read_table('./model/simple/actor_loss.txt',header=None, delim_whitespace=True)
data22 = data2.values
df_actor_loss = pd.DataFrame(data22)

plt.figure("actor loss")
plt.plot(range(len(actor_loss_data)), df_actor_loss[0].rolling(50, min_periods=1).mean())
plt.xlabel('time_step ')
plt.ylabel('actor_loss smooth')
plt.savefig('./model/simple' + '/actor_loss_smooth.pdf',  bbox_inches="tight", format='pdf')

reward_data = np.loadtxt('./model/simple/reward.txt', dtype=np.float32, delimiter=',')
data3 = pd.read_table('./model/simple/reward.txt',header=None, delim_whitespace=True)
data32 = data3.values
df_reward = pd.DataFrame(data32)

plt.figure("reward")
plt.plot(range(len(reward_data)), df_reward[0].rolling(50, min_periods=1).mean())
plt.xlabel('episode * ' + str(1))
plt.ylabel('reward smooth')
plt.savefig('./model/simple' + '/reward_smooth.pdf',  bbox_inches="tight", format='pdf')

comp_data = np.loadtxt('./model/simple/comp_allocation.txt', dtype=np.float32, delimiter=',')
data4 = pd.read_table('./model/simple/comp_allocation.txt',header=None, delim_whitespace=True)
data42 = data4.values
df_comp = pd.DataFrame(data42)

plt.figure("computing allocation")
plt.plot(range(len(comp_data)), df_comp[0].rolling(50, min_periods=1).mean())
plt.xlabel('time_step ')
plt.ylabel('coomp_allocation smooth')
plt.savefig('./model/simple' + '/comp_allocation_smooth.pdf',  bbox_inches="tight", format='pdf')

band_data = np.loadtxt('./model/simple/band_allocation.txt', dtype=np.float32, delimiter=',')
data5 = pd.read_table('./model/simple/band_allocation.txt',header=None, delim_whitespace=True)
data52 = data5.values
df_band = pd.DataFrame(data52)

plt.figure("spectrum allocation")
plt.plot(range(len(band_data)), df_band[0].rolling(50, min_periods=1).mean())
plt.xlabel('time_step ')
plt.ylabel('band smooth')
plt.savefig('./model/simple' + '/band_smooth.pdf',  bbox_inches="tight", format='pdf')

hit_rate_data = np.loadtxt('./model/simple/hit_rate.txt', dtype=np.float32, delimiter=',')
data6 = pd.read_table('./model/simple/hit_rate.txt',header=None, delim_whitespace=True)
data62 = data6.values
df_hit_rate = pd.DataFrame(data62)

plt.figure("hit rate")
plt.plot(range(len(hit_rate_data)), df_hit_rate[0].rolling(50, min_periods=1).mean())
plt.xlabel('time_step ')
plt.ylabel('hit_rate smooth')
plt.savefig('./model/simple' + '/hit_rate_smooth.pdf',  bbox_inches="tight", format='pdf')

plt.show()
