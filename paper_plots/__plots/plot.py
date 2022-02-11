from matplotlib import rc_file
rc_file(r'C:\Users\boettner\Google Drive\Uni\Paper\AR(1) Paper\__plots\settings.rc')
import matplotlib.pyplot as plt
import numpy as np

orange = '#D55E00'
purple = '#330066'

plt.close("all")
fig = plt.figure()
# =============================================================================
num = 5
lm = np.load(f"roc/e{num}/lm_positive_roc.npy"); lm[-1,:] = np.array([1,1,1,1,1])
lm_n = np.load(f"roc/e{num}/lm_positive_roc_n.npy"); lm_n[-1,:] = np.array([1,1,1,1,1])

ax1 = fig.add_subplot(121)
ax1.set_title("White Noise", fontsize=20)
ax1.plot(lm_n[:,0],lm_n[:,4], label = r"Adjusted Parameter $\varphi_a$", color = orange, marker = 'o', markevery = np.arange(0,len(lm),5))
ax1.plot(lm_n[:,0],lm_n[:,3], label = r"AR(1) Parameter $\varphi$", color = "grey")
ax1.plot(lm_n[:,0],lm_n[:,2], label = r"Lag-1 Autocorrelation $\alpha_1$", color = "grey", linestyle="--")
ax1.plot(lm_n[:,0],lm_n[:,1], label = r"Variance $\sigma^2$", color = "grey", linestyle="-.")
ax1.plot(lm_n[:,0],lm_n[:,0], color="grey", linestyle= ":", label = "Null Model")
ax1.spines["right"].set_visible(False)

ax2 = fig.add_subplot(122)
ax2.set_title("Non-Stationary AR(1) Noise", fontsize=20)
ax2.plot(lm[:,0],lm[:,4], label = r"Adjusted Parameter $\varphi_a$", color = orange, marker = 'o', markevery = np.arange(0,len(lm),5))
ax2.plot(lm[:,0],lm[:,3], label = r"AR(1) Parameter $\varphi$", color = "grey")
ax2.plot(lm[:,0],lm[:,2], label = r"Lag-1 Autocorrelation $\alpha_1$", color = "grey", linestyle="--")
ax2.plot(lm[:,0],lm[:,1], label = r"Variance $\sigma^2$", color = "grey", linestyle="-.")
ax2.plot(lm[:,0],lm[:,0], color="grey", linestyle= ":", label = "Null Model")
ax2.legend()
ax2.spines["left"].set_visible(False)
ax2.tick_params(left=False, labelleft=False)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Significance Threshold")
plt.ylabel("Proportion of Significant Trends", labelpad = 10)

# =============================================================================
fig.suptitle("Double-Well Model", fontsize=24, x=0.55)
fig.tight_layout(pad=0.1)  # Make the figure use all available whitespace
plt.subplots_adjust(top=0.944,
bottom=0.116,
left=0.079,
right=0.983,
hspace=0.1,
wspace=0.1)