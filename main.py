from astropy.io import fits
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv

#---------------------------------------

def main():
    """
    Main function.
    """

    csvfilepath = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/CatWISE Planemos.csv"
    fileanalysis(None, photometry='allwise', annotate=False)
    copycsvfilepath = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/Copy of CatWISE Planemos.csv"
    flagcsv = flagcatwise(copycsvfilepath)

    header = ["Name", "Flag"]


    with open('CatWise_Flag.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        
        # write multiple rows
        writer.writerows(flagcsv)

    return

#---------------------------------------

def cmdplot(m, l, t, y, overplot, photometry, annotate):
    """
    This code plots a CMD of the data inputted.
    Inputs:
    m                Array containing J mag, J - K, (or W1/W1-W2) and corresponding errors for M-type brown dwarfs.
    l                Array containing J mag, J - K, (or W1/W1-W2) and corresponding errors for L-type brown dwarfs.
    t                Array containing J mag, J - K, (or W1/W1-W2) and corresponding errors for T-type brown dwarfs.
    y                Array containing J mag, J - K, (or W1/W1-W2) and corresponding errors for Y-type brown dwarfs.
    overplot         Array containing J mag, J - K, (or W1/W1-W2) and corresponding errors for inputted brown dwarfs.
    photometry       Photometry type for plotting -- "MKO", "2MASS", or "ALLWISE". 

    Outputs:
    None
    
    """
    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plotting data
    plt.errorbar(m[:,1], m[:,0], m[:,3], m[:,2], color='navy', ecolor ='lightsteelblue', marker='o', ls='none', mec='lightskyblue', ms=9, mew=0.5, alpha=0.7, label='M - Type')
    plt.errorbar(l[:,1], l[:,0], l[:,3], l[:,2], color='darkgreen', ecolor='lightgreen', marker='^', ls='none', mec='honeydew', ms=10, mew=0.4, alpha=0.7, label='L - Type')
    plt.errorbar(t[:,1], t[:,0], t[:,3], t[:,2], color='maroon', ecolor='lightcoral', marker='d', ls='none', mec='pink', ms=10, mew=0.5, alpha=0.7, label='T - Type')
    plt.errorbar(y[:,1], y[:,0], y[:,3], y[:,2], color='indigo', ecolor='thistle', marker='s', ls='none', mec='plum', ms=9, mew=0.5, alpha=0.7, label='Y - Type')
    
    if len(overplot) == 0:
        placeholder = 1
    else:
        plt.errorbar(overplot[:,1].astype('float64'), overplot[:,0].astype('float64'), overplot[:,3].astype('float64'), overplot[:,2].astype('float64'), color='orangered', marker='*', ecolor='sandybrown', ls='none', mec='peachpuff', ms=15, mew=0.5,alpha=0.6, label= "Overplotted Data")
        # Label the overplotted data points
        nameover = overplot[:,4]
        x = overplot[:,1].astype('float64')
        y = overplot[:,0].astype('float64')
        if annotate == True:
            for i, txt in enumerate(nameover):
                ax.annotate(txt, (x[i],y[i]), xytext = (x[i]-0.1, y[i]+0.25), arrowprops=dict(color="orangered", arrowstyle="-"), fontsize=12, fontweight='bold', color="orangered")
    
    if photometry == "ALLWISE":
        plt.legend(loc="upper right", prop={'size': 16})
    else:
        plt.legend(loc="upper left", prop={'size': 16})
    
    # Invert y-axis
    ax.invert_yaxis()

    # Minor axes
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Set axes colours
    ax.tick_params(which="major", size=9, labelsize=21, width=3, direction='in', top=True, right=True)
    ax.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in', top=True, right=True)

    # Axes widths
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    # Axis titles
    if photometry == "ALLWISE":
        plt.xlabel("$\it{W1 - W2}$", fontsize=28)
        plt.ylabel("M$\it{_{W1}}$", fontsize=28)
    else:
        plt.xlabel("$J - K$", fontsize=28)
        plt.ylabel("M$_J$", fontsize=28)

    # Save figure
    savename = "browndwarfCMD" + photometry
    plt.savefig(savename)
    
    return


#---------------------------------------


def fileanalysis(inputfilepath=None, photometry="MKO", annotate=True):
    """
    This function takes in files with data to plot, breaks it into the proper catagories, 
    then calls the plotting function.

    Inputs:
    inputfilepath         File path to csv with data for overplotting.
    photometry            Photometry data to use in plotting -- "MKO", "2MASS" or "ALLWISE". "MKO" is the defult.
    annotate              Boolean. If True, overplot data will be annotated with object name in plot. True is the defult.

    Outputs:
    None
    
    """
    
    # To allow user to choose MKO or 2MASS data
    photometry = photometry.upper()
    
    # For J-K photometry
    if (photometry == "MKO") or (photometry == "2MASS"): 

        # Read data file
        with fits.open("vlm-plx-all.fits") as browndwarflist:
            data = browndwarflist[1].data
            #hdr = browndwarflist[1].header
            
        # Setting variables from fits file
        objnames = data["NAME"]
        spec = data["ISPTSTR"]

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
        elif photometry == "ALLWISE":
            j = np.array([])
            jerror = np.array([])
            k = np.array([])
            kerror = np.array([])
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
            bestdata = np.loadtxt("bestmatcheddata.csv", dtype=str, delimiter=",", skiprows=1)

            bestjmag = np.array(bestdata[:,10])
            bestjmag = np.where(bestjmag=="", np.nan, bestjmag).astype('float64')

            bestkmag = np.array(bestdata[:,14])
            bestkmag = np.where(bestkmag=="", np.nan, bestkmag).astype('float64')

            bestjerror = np.array(bestdata[:, 11])
            bestjerror = np.where(bestjerror=="", np.nan, bestjerror).astype('float64')

            bestkerror = np.array(bestdata[:, 15])
            bestkerror = np.where(bestkerror=="", np.nan, bestkerror).astype('float64')

            bestjkmag = bestjmag - bestkmag
            bestjkerror = np.sqrt(bestjerror**2 + bestkerror**2)

            bestspectype = np.array(bestdata[:,9])

            
            # Sort data into spectral types
            for ispec in range(len(bestspectype)):
                if "M" in bestspectype[ispec]:
                    m_jmag.append(bestjmag[ispec])
                    m_jkmag.append(bestjkmag[ispec])
                    m_jmagerror.append(bestjerror[ispec])
                    m_jkmagerror.append(bestjkerror[ispec])
                    m_spec.append(bestspectype[ispec])
                    #print(spectype[ispec])

                elif "L" in bestspectype[ispec]:
                    l_jmag.append(bestjmag[ispec])
                    l_jkmag.append(bestjkmag[ispec])
                    l_jmagerror.append(bestjerror[ispec])
                    l_jkmagerror.append(bestjkerror[ispec])
                    #print(spectype[ispec])

                elif "T" in bestspectype[ispec]:
                    t_jmag.append(bestjmag[ispec])
                    t_jkmag.append(bestjkmag[ispec])
                    t_jmagerror.append(bestjerror[ispec])
                    t_jkmagerror.append(bestjkerror[ispec])
                    #print(spectype[ispec])

                elif "Y" in bestspectype[ispec]:
                    y_jmag.append(bestjmag[ispec])
                    y_jkmag.append(bestjkmag[ispec])
                    y_jmagerror.append(bestjerror[ispec])
                    y_jkmagerror.append(bestjkerror[ispec])
                    #print(spectype[ispec])

                else:
                    null_jmag.append(bestjmag[ispec])
                    null_jkmag.append(bestjkmag[ispec])
                    null_jmagerror.append(bestjerror[ispec])
                    null_jkmagerror.append(bestjkerror[ispec])
            
        
        # Create arrays for plotting
        m = np.vstack((m_jmag, m_jkmag, m_jmagerror, m_jkmagerror)).T
        l = np.vstack((l_jmag, l_jkmag, l_jmagerror, l_jkmagerror)).T
        t = np.vstack((t_jmag, t_jkmag, t_jmagerror, t_jkmagerror)).T
        y = np.vstack((y_jmag, y_jkmag, y_jmagerror, y_jkmagerror)).T


    # Add allWISE data if ALLWISE
    elif photometry == "ALLWISE":
        allwisedata = np.loadtxt("allwisematcheddata.csv", dtype=str, delimiter=",", skiprows=1)

        w1mag = np.array(allwisedata[:,10]).astype('float64')
        w2mag = np.array(allwisedata[:,12]).astype('float64')
        ew1mag = np.array(allwisedata[:,11])
        ew1mag = np.where(ew1mag=="", 0, ew1mag).astype('float64')
        ew2mag = np.array(allwisedata[:,13]).astype('float64')

        w12mag = w1mag - w2mag
        ew12mag = np.sqrt(ew1mag**2 + ew2mag**2)

        spectype = np.array(allwisedata[:,9])

        m_w1mag = []
        l_w1mag = []
        t_w1mag = []
        y_w1mag = []
        null_w1mag = []
        m_w12mag = []
        l_w12mag = []
        t_w12mag = []
        y_w12mag = []
        null_w12mag = []
        m_w1magerror = []
        l_w1magerror = []
        t_w1magerror = []
        y_w1magerror = []
        null_w1magerror = []
        m_w12magerror = []
        l_w12magerror = []
        t_w12magerror = []
        y_w12magerror = []
        null_w12magerror = []
        m_spec = []

        # Sort data into spectral types
        for ispec in range(len(spectype)):
            if "M" in spectype[ispec]:
                m_w1mag.append(w1mag[ispec])
                m_w12mag.append(w12mag[ispec])
                m_w1magerror.append(ew1mag[ispec])
                m_w12magerror.append(ew12mag[ispec])
                m_spec.append(spectype[ispec])
                #print(spectype[ispec])

            elif "L" in spectype[ispec]:
                l_w1mag.append(w1mag[ispec])
                l_w12mag.append(w12mag[ispec])
                l_w1magerror.append(ew1mag[ispec])
                l_w12magerror.append(ew12mag[ispec])
                #print(spectype[ispec])

            elif "T" in spectype[ispec]:
                t_w1mag.append(w1mag[ispec])
                t_w12mag.append(w12mag[ispec])
                t_w1magerror.append(ew1mag[ispec])
                t_w12magerror.append(ew12mag[ispec])
                #print(spectype[ispec])

            elif "Y" in spectype[ispec]:
                y_w1mag.append(w1mag[ispec])
                y_w12mag.append(w12mag[ispec])
                y_w1magerror.append(ew1mag[ispec])
                y_w12magerror.append(ew12mag[ispec])
                #print(spectype[ispec])

            else:
                null_w1mag.append(w1mag[ispec])
                null_w12mag.append(w12mag[ispec])
                null_w1magerror.append(ew1mag[ispec])
                null_w12magerror.append(ew12mag[ispec])


        # Create arrays for plotting
        m = np.vstack((m_w1mag, m_w12mag, m_w1magerror, m_w12magerror)).T
        l = np.vstack((l_w1mag, l_w12mag, l_w1magerror, l_w12magerror)).T
        t = np.vstack((t_w1mag, t_w12mag, t_w1magerror, t_w12magerror)).T
        y = np.vstack((y_w1mag, y_w12mag, y_w1magerror, y_w12magerror)).T


    else:
        print("Invalid photometry entered")

    # Overplotting data
    if inputfilepath == None:
        overplot = np.array([]).astype("float64")
    else:
        overplotdata = np.loadtxt(inputfilepath, dtype=str, delimiter=",", skiprows=1)

        namesover = np.array(overplotdata[:, 0])

        if photometry == "ALLWISE":
            w1magover = np.array(overplotdata[:, 3]).astype("float64")
            w12magover = np.array(overplotdata[:, 4]).astype("float64")
            ew1magover = np.zeros(len(w1magover))
            ew12magover = np.zeros(len(w12magover))
            overplot = np.vstack((w1magover, w12magover, ew1magover, ew12magover, namesover)).T


        elif photometry == "MKO":
            jmagover = np.array(overplotdata[:, 53])
            kmagover = np.array(overplotdata[:, 56])
            ejmagover = np.array(overplotdata[:, 54])
            ekmagover = np.array(overplotdata[:, 57])

            jkmagover = jmagover - kmagover
            ejkmagover = np.sqrt(ejmagover**2 + ekmagover**2)
            overplot = np.vstack((jmagover, jkmagover, ejmagover, ejkmagover, namesover)).T


        elif photometry == "2MASS":
            jmagover = np.array(overplotdata[:, 53])
            kmagover = np.array(overplotdata[:, 56])
            ejmagover = np.array(overplotdata[:, 54])
            ekmagover = np.array(overplotdata[:, 57])

            jkmagover = jmagover - kmagover
            ejkmagover = np.sqrt(ejmagover**2 + ekmagover**2)
            overplot = np.vstack((jmagover, jkmagover, ejmagover, ejkmagover, namesover)).T



    cmdplot(m, l, t, y, overplot, photometry, annotate)
   

    return m, l, t, y, overplot


#----------------------------------

def flagcatwise(csvfilepath):
    """
    
    """

    
    m, l, t, y, overplot = fileanalysis(csvfilepath, photometry='allwise', annotate=False)


    w1mag = np.hstack((m[:,0], l[:,0], t[:,0]))
    w12mag = np.hstack((m[:,1], l[:,1], t[:,1]))

    no_outw1 = np.array([])
    no_outw12 = np.array([])


    for i in range(len(w1mag)):
        if not ((w1mag[i] > 16.0) and (w12mag[i] < 1.6)):
            no_outw1 = np.append(no_outw1, w1mag[i])
            no_outw12 = np.append(no_outw12, w12mag[i])

    binnum = int((np.max(w12mag)-np.min(w12mag))//0.1)
    binhwid = 0.2
    bincen = 0.1

    allbinmaxes = np.empty((0, 3))

    for i in range(binnum):
        binmin = bincen - binhwid
        binmax = bincen + binhwid
        bincen+=0.1
        thebin = np.empty((0))
        
        for j in range(len(no_outw12)):
            if (no_outw12[j] >= binmin) and (no_outw12[j] <= binmax):
                thebin = np.append(thebin, no_outw1[j])
        
        if len(thebin) == 0:
            binmaxval = 0
        else:
            binmaxval = np.max(thebin)
            
        allbinmaxes = np.append(allbinmaxes, np.array([[binmaxval, binmin, binmax]]), axis=0)

    x = (allbinmaxes[:, 1] + allbinmaxes[:,2])/2
    y = allbinmaxes[:, 0]
    x = np.linspace(0.1, 3.7, len(y))
    m, b = np.polyfit(x, y, 1)
    #m = slope, b = intercept

    overw1 = overplot[:,0].astype("float64")
    overw12 = overplot[:,1].astype("float64")
    overnames = overplot[:,4]
    forcatwise = np.empty((0, 2))
            
    flag = overw1 > m*overw12+b
    flagw1 = overw1[flag]
    flagw12 = overw12[flag]
    flagnames = overnames[flag]

    newflgw1 = []
    newflgw12 = []
    for i in range(25):
        for j in range(len(flagw1)):
            if flagw12[j] < 2.3:
                flagx = "x"
                newflgw1.append(flagw1[j])
                newflgw12.append(flagw12[j])
        
                forcatwise = np.append(forcatwise, np.array([[flagnames[j], flagx]]), axis=0)

    forcatwise = np.unique(forcatwise, axis=0)
    newforcatwise = forcatwise

    for i in range(len(overnames)):
        if overnames[i] not in forcatwise[:,0]:
            newforcatwise = np.append(newforcatwise, np.array([[overnames[i], ""]]), axis=0)
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.scatter(w12mag, w1mag, label = "Disregarded Best Data")
    plt.scatter(no_outw12, no_outw1, label = "Best et. al. (2020) Data")
    plt.scatter(newflgw12, newflgw1, c="k", marker="*", label = "Flagged CatWISE Data")
    plt.plot(x, m*x + b, c="green", label = "Bin Min Best Fit")
    plt.legend()
    ax.invert_yaxis()
    plt.xlabel("W1 - W2")
    plt.ylabel("W1")
    plt.savefig("cmd_flag")

    return newforcatwise


if __name__ == "__main__":
    main()