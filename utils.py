import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

from scipy import signal
    
def zplane_ba(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k

def zplane_zp(zeros, poles):
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # Plot the zeros and set marker properties    
    t1 = plt.plot(zeros.real, zeros.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(poles.real, poles.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

def freq_analysis(poles, zeros, k, fs, n):


    plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 2, 1)

    # 1. plot the poles and zeros
    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)
    # Plot the zeros and set marker properties    
    ax.plot(zeros.real, zeros.imag, 'go', ms=10)
    # ax.setp( t1, markersize=10.0, markeredgewidth=1.0,
    #           markeredgecolor='k', markerfacecolor='g')
    # Plot the poles and set marker properties
    ax.plot(poles.real, poles.imag, 'rx', ms=10)
    # ax.setp( t2, markersize=12.0, markeredgewidth=3.0,
    #           markeredgecolor='r', markerfacecolor='r')
    ax.set_ylabel("Imaginary", color='b')
    ax.set_xlabel("Real", color='b')

    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title("Poles and Zeros")
    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    plt.subplot(1, 2, 2)
    # 2. plot the impulse response
    dt = 1/fs
    t, y = signal.dimpulse((zeros, poles, k, dt), n=n)
    y = np.squeeze(y)
    plt.plot(t, y)
    plt.xlim([0, n*dt])
    plt.xlabel("Time (s)", color='b')
    plt.ylabel("Amplitude", color='b')
    plt.title("Impulse Response")

    # 3. plot the magnitude response
    yF = np.fft.fft(y)
    yF = np.fft.fftshift(yF)
    yF_mag = 20*np.log10(np.abs(yF))
    yF_ang = np.angle(yF)

    freq = np.fft.fftfreq(len(yF), dt)
    freq = np.fft.fftshift(freq)

    # 1x2 Plot of Magnitude and Phase
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(freq[len(freq)//2:], yF_mag[len(freq)//2:])
    ax1.set_xlabel("Frequency (Hz)", color='b')
    ax1.set_ylabel("Magnitude (dB)", color='b')
    ax1.set_title("Magnitude Response")
    ax2.plot(freq, np.unwrap(yF_ang))
    ax2.set_xlabel("Frequency (Hz)", color='b')
    ax2.set_ylabel("Phase (rad)", color='b')
    ax2.set_title("Phase Response Unwrapped")

    # plt.tight_layout()