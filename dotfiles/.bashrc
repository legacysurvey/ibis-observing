export PS1='\u@\h:\w\n> '

alias e='emacs -nw'
alias t='tail -f'
alias python='python3'

# python line_profiler
#export LINE_PROFILE=1

# observer4
export PS1CAT_DIR=/data/declsp/PS1/chunks-qz-star-v2/

export GAIA_CAT_DIR=/data/declsp/gaia-dr3/healpix/
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=~/SFD

# For Astrometry.net build: static linking
#export WCSLIB_LIB=~/wcslib/lib/libwcs.a
#export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:~/wcslib/lib/pkgconfig

# libav video format converter
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/libav/lib
#export PATH=${PATH}:~/libav/bin

export PATH=${PATH}:~/postgresql-17.2/bin

export PATH=${PATH}:$HOME/python-libs/astrometry/util:$HOME/python-libs/astrometry/solver:$HOME/python-libs/astrometry/plot

export PYTHONPATH=${PYTHONPATH}:~/python-libs
export PATH=${PATH}:~/python-libs/bin

export PATH=${PATH}:~/bin

# We don't end up needing mamba/conda/etc

# added by Miniconda3 installer
# export PATH="/user/declsp/miniconda3/bin:$PATH"  # commented out by conda initialize

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home2/declsp/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home2/declsp/mambaforge/etc/profile.d/conda.sh" ]; then
#        . "/home2/declsp/mambaforge/etc/profile.d/conda.sh"
#    else
#        export PATH="/home2/declsp/mambaforge/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<

#source activate /home2/declsp/mambaforge
