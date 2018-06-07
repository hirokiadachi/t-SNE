import matplotlib as mpl

def layout():
    c_cycle = ("#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#f1c40f", "#ecf0f1", "#34495e",
               "#446cb3", "#d24d57", "#27ae60", "#663399", "#f7ca18", "#bdc3c7", "#2c3e50")
    mpl.rc('font', size=10)
    mpl.rc('lines', linewidth=2, color="#2c3e50")
    mpl.rc('patch', linewidth=0, facecolor="none", edgecolor="none")
    mpl.rc('text', color='#2c3e50')
    # mpl.rc('axes', facecolor='none', edgecolor="none", titlesize=20, labelsize=15, color_cycle=c_cycle, grid=False)
    mpl.rc('axes', titlesize=15, labelsize=12)
    mpl.rc('xtick.major', size=10, width=0)
    mpl.rc('ytick.major', size=10, width=0)
    mpl.rc('xtick.minor', size=10, width=0)
    mpl.rc('ytick.minor', size=10, width=0)
    mpl.rc('ytick', direction="out")
    mpl.rc('grid', color='#c0392b', alpha=0.5, linewidth=1)
    mpl.rc('legend', fontsize=25, markerscale=1, labelspacing=0.2, frameon=True, fancybox=True,
           handlelength=0.1, handleheight=0.5, scatterpoints=1, facecolor="#eeeeee")
    mpl.rc('figure', figsize=(10, 6), dpi=224, facecolor="none", edgecolor="none")
    mpl.rc('savefig', dpi=1500)

