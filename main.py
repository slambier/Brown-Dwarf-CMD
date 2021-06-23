from astropy.io import fits
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


#---------------------------------------

#def cmdplot(m, l, t, y, overplot):
def cmdplot(m, l, t, y):
    """
    This code plots a CMD of the data inputted. 
    
    """
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    ax.invert_yaxis()
    plt.errorbar(m[:,1], m[:,0], m[:,3], m[:,2], color='navy', ecolor ='lightsteelblue', marker='o', ls='none', mec='lightskyblue', ms=9, mew=0.5, alpha=0.7, label='M - Type')
    plt.errorbar(l[:,1], l[:,0], l[:,3], l[:,2], color='darkgreen', ecolor='lightgreen', marker='^', ls='none', mec='honeydew', ms=10, mew=0.4, alpha=0.7, label='L - Type')
    plt.errorbar(t[:,1], t[:,0], t[:,3], t[:,2], color='maroon', ecolor='lightcoral', marker='d', ls='none', mec='pink', ms=10, mew=0.5, alpha=0.7, label='T - Type')
    plt.errorbar(y[:,1], y[:,0], y[:,3], y[:,2], color='indigo', ecolor='thistle', marker='s', ls='none', mec='plum', ms=9, mew=0.5, alpha=0.7, label='Y - Type')
    #plt.errorbar(null_jkmag, null_jmag, null_jkmagerror, null_jmagerror, color='orangered', marker='*', ecolor='sandybrown', ls='none', mec='peachpuff', ms=24, mew=0.5,alpha=0.7, label='Unknown Type')
    plt.legend(loc="upper left", prop={'size': 16})

    # Minor axes
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Set axes colours
    ax.tick_params(which="major", size=9, labelsize=21, width=3, direction='in', top=True, right=True)
    ax.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in', top=True, right=True)

    # Axes widths
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    plt.xlabel("$J - K$", fontsize=28)
    plt.ylabel("M$_J$", fontsize=28)

    plt.savefig("browndwarfCMD")
    
    return


#---------------------------------------



#def fileanalysis(inputfilepath, photometry = "MKO"):
def fileanalysis(photometry="MKO"):
    """
    This function takes in files with data to plot, breaks it into the proper catagories, 
    then calls the plotting function.
    
    """
    # Read data file
    with fits.open("vlm-plx-all.fits") as browndwarflist:
        data = browndwarflist[1].data
        hdr = browndwarflist[1].header
        
    # Setting variables from fits file
    objnames = data["NAME"]
    spec = data["ISPTSTR"]
    
    # To allow user to choose MKO or 2MASS data
    photometry = photometry.upper()
    
    if photometry == "MKO":
        j = data["JMAG"]
        jerror = data["EJMAG"]
        k = data["KMAG"]
        kerror = data["EKMAG"]
    elif photometry == "2MASS":
        j = data["J2MAG"]
        jerror = data["EJ2MAG"]
        k = data["K2MAG"]
        kerror = data["EK2MAG"]
    else:
        print("Invalid photometry entered.")
        exit()

    # Read csv of binary stars to remove
    binarycsv = np.loadtxt("binaryname.csv", skiprows=2, dtype=str)

    newobjnames = np.array([])
    spectype = np.array([])
    jmag = np.array([])
    jmagerror = np.array([])
    kmag = np.array([])
    kmagerror = np.array([])

    for i in range(len(objnames)):
        if objnames[i] not in binarycsv:
            newobjnames = np.append(newobjnames, objnames[i])
            spectype= np.append(spectype, spec[i])
            jmag = np.append(jmag, j[i])
            jmagerror = np.append(jmagerror, jerror[i])
            kmag = np.append(kmag, k[i])
            kmagerror = np.append(kmagerror, kerror[i])
            
    # Calculate J-K and J-K errorbars
    jkmag = jmag - kmag
    jkmagerror = np.sqrt(jmagerror**2 + kmagerror**2)


    # Remove subdwarfs
    jmagd = np.array([])
    jmagerrord = np.array([])
    jkmagd = np.array([])
    jkmagerrord = np.array([])
    spectyped = np.array([])

    for i in range(len(spectype)):
        if "sd" not in spectype[i]:
            jmagd = np.append(jmagd, jmag[i])
            jmagerrord = np.append(jmagerrord, jmagerror[i])
            jkmagd= np.append(jkmagd, jkmag[i])
            jkmagerrord= np.append(jkmagerrord, jkmagerror[i])
            spectyped = np.append(spectyped, spectype[i])
            #print(spectype[i])

    # Create arrays for different spec types
    m_jmag = []
    l_jmag = []
    t_jmag = []
    y_jmag = []
    null_jmag = []
    m_jkmag = []
    l_jkmag = []
    t_jkmag = []
    y_jkmag = []
    null_jkmag = []
    m_jmagerror = []
    l_jmagerror = []
    t_jmagerror = []
    y_jmagerror = []
    null_jmagerror = []
    m_jkmagerror = []
    l_jkmagerror = []
    t_jkmagerror = []
    y_jkmagerror = []
    null_jkmagerror = []
    m_spec = []

    # Sort data into spectral types
    for ispec in range(len(spectyped)):
        if "M" in spectyped[ispec]:
            m_jmag.append(jmagd[ispec])
            m_jkmag.append(jkmagd[ispec])
            m_jmagerror.append(jmagerrord[ispec])
            m_jkmagerror.append(jkmagerrord[ispec])
            m_spec.append(spectyped[ispec])
            #print(spectype[ispec])

        elif "L" in spectyped[ispec]:
            l_jmag.append(jmagd[ispec])
            l_jkmag.append(jkmagd[ispec])
            l_jmagerror.append(jmagerrord[ispec])
            l_jkmagerror.append(jkmagerrord[ispec])
            #print(spectype[ispec])

        elif "T" in spectyped[ispec]:
            t_jmag.append(jmagd[ispec])
            t_jkmag.append(jkmagd[ispec])
            t_jmagerror.append(jmagerrord[ispec])
            t_jkmagerror.append(jkmagerrord[ispec])
            #print(spectype[ispec])

        elif "Y" in spectyped[ispec]:
            y_jmag.append(jmagd[ispec])
            y_jkmag.append(jkmagd[ispec])
            y_jmagerror.append(jmagerrord[ispec])
            y_jkmagerror.append(jkmagerrord[ispec])
            #print(spectype[ispec])

        else:
            null_jmag.append(jmagd[ispec])
            null_jkmag.append(jkmagd[ispec])
            null_jmagerror.append(jmagerrord[ispec])
            null_jkmagerror.append(jkmagerrord[ispec])

    # Add Best data if 2MASS
    if photometry == "2MASS":
        bestdata = np.loadtxt("best95list.txt", dtype=str, delimiter=";", skiprows=2)

        bestjmag = np.array(bestdata[:,7])
        bestjmag = np.where(bestjmag=="null", np.nan, bestjmag).astype('float64')

        bestkmag = np.array(bestdata[:,8])
        bestkmag = np.where(bestkmag=="null", np.nan, bestkmag).astype('float64')

        bestjkmag = bestjmag - bestkmag

        bestspectype = np.array(bestdata[:,9])

        bestplusj = np.array([])
        bestplusjk= np.array([])
        bestplusspec = np.array([])

        for ispec in range(len(bestspectype)):
            if '+' not in bestspectype[ispec]:
                bestplusj = np.append(bestplusj, bestjmag[ispec])
                bestplusjk = np.append(bestplusjk, bestjkmag[ispec])
                bestplusspec = np.append(bestplusspec, bestspectype[ispec])

        # comment is red
        bestjerror = np.zeros(len(bestplusj))
        bestjkerror = np.zeros(len(bestplusj))
        
        # Sort data into spectral types
        for ispec in range(len(bestplusspec)):
            if "M" in bestplusspec[ispec]:
                m_jmag.append(bestplusj[ispec])
                m_jkmag.append(bestplusjk[ispec])
                m_jmagerror.append(bestjerror[ispec])
                m_jkmagerror.append(bestjkerror[ispec])
                m_spec.append(bestplusspec[ispec])
                #print(spectype[ispec])

            elif "L" in bestplusspec[ispec]:
                l_jmag.append(bestplusj[ispec])
                l_jkmag.append(bestplusjk[ispec])
                l_jmagerror.append(bestjerror[ispec])
                l_jkmagerror.append(bestjkerror[ispec])
                #print(spectype[ispec])

            elif "T" in bestplusspec[ispec]:
                t_jmag.append(bestplusj[ispec])
                t_jkmag.append(bestplusjk[ispec])
                t_jmagerror.append(bestjerror[ispec])
                t_jkmagerror.append(bestjkerror[ispec])
                #print(spectype[ispec])

            elif "Y" in bestplusspec[ispec]:
                y_jmag.append(bestplusj[ispec])
                y_jkmag.append(bestplusjk[ispec])
                y_jmagerror.append(bestjerror[ispec])
                y_jkmagerror.append(bestjkerror[ispec])
                #print(spectype[ispec])

            else:
                null_jmag.append(bestplusj[ispec])
                null_jkmag.append(bestplusjk[ispec])
                null_jmagerror.append(bestjerror[ispec])
                null_jkmagerror.append(bestjkerror[ispec])
    
    m = np.vstack((m_jmag, m_jkmag, m_jmagerror, m_jkmagerror)).T
    l = np.vstack((l_jmag, l_jkmag, l_jmagerror, l_jkmagerror)).T
    t = np.vstack((t_jmag, t_jkmag, t_jmagerror, t_jkmagerror)).T
    y = np.vstack((y_jmag, y_jkmag, y_jmagerror, y_jkmagerror)).T
    
    #cmdplot(m, l, t, y, overplot)
    cmdplot(m, l, t, y)

    return


fileanalysis()