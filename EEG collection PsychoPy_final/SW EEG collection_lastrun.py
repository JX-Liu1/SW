#!/usr/bin/env python
# -*- coding: utf-8 -*-


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'SW EEG collection'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/josh/Desktop/Jiaxing Liu SW/EEG collection PsychoPy_final/SW EEG collection_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('welcome_continue') is None:
        # initialise welcome_continue
        welcome_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcome_continue',
        )
    if deviceManager.getDevice('intro_continue') is None:
        # initialise intro_continue
        intro_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_continue',
        )
    if deviceManager.getDevice('resting_state_end_continue') is None:
        # initialise resting_state_end_continue
        resting_state_end_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='resting_state_end_continue',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Welcome to the experiment!\n\nTap spacebar to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welcome_continue = keyboard.Keyboard(deviceName='welcome_continue')
    
    # --- Initialize components for Routine "intro" ---
    intro_text = visual.TextStim(win=win, name='intro_text',
        text="Welcome to the Study!\n\nIn this experiment, we’ll first collect your resting-state EEG data for 30 seconds. This means you’ll relax and let us record your brain activity.\n\nPlease adjust to a comfortable position.\n\nWe will now begin collecting your resting-state EEG data. \n\nWhen you're ready, press the spacebar to start.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_continue = keyboard.Keyboard(deviceName='intro_continue')
    
    # --- Initialize components for Routine "resting_state_EEG" ---
    cross_sign = visual.ShapeStim(
        win=win, name='cross_sign', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "resting_state_end" ---
    resting_state_end_text = visual.TextStim(win=win, name='resting_state_end_text',
        text="This is the end of resting-state EEG collection.\n\nNext we will begin ERP data collection. \n\nHere’s what will happen in each section of the study:\n\nPreparation Time: You will see a cross on the screen for 6 seconds. Use this time to relax.\n\nPicture Viewing: A picture will then appear for 4 seconds. Please remain still and avoid eye blinking or body movement during this time.\n\nReport: After viewing the picture, you will indicate whether you like the content of the picture or not.\n\nPlease adjust to a comfortable position.\n\nWhen you're ready, press the spacebar to start.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    resting_state_end_continue = keyboard.Keyboard(deviceName='resting_state_end_continue')
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "huangshan" ---
    huangshan_image = visual.ImageStim(
        win=win,
        name='huangshan_image', 
        image='images/eaaedf0b69306fdc1bd42256840db8e5.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "huangshan_souvenir" ---
    huangshan_souvenir_image = visual.ImageStim(
        win=win,
        name='huangshan_souvenir_image', 
        image='images/1485282ca2d63b758886ce8466043d65.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "forbidden_city" ---
    forbidden_city_image = visual.ImageStim(
        win=win,
        name='forbidden_city_image', 
        image='images/forbidden city.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "forbidden_city_souvenir" ---
    forbidden_city_souvenir_image = visual.ImageStim(
        win=win,
        name='forbidden_city_souvenir_image', 
        image='images/97843262-3039-42cc-a201-af4397613a97.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "suzhou" ---
    suzhou_image = visual.ImageStim(
        win=win,
        name='suzhou_image', 
        image='images/477db1189aa76e7022194d8065e80485.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "suzhou_souvenir" ---
    suzhou_souvenir_image = visual.ImageStim(
        win=win,
        name='suzhou_souvenir_image', 
        image='images/99f97ea5b779197516a424e547976194.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "shanghai" ---
    shanghai_image = visual.ImageStim(
        win=win,
        name='shanghai_image', 
        image='images/fcaa1942084bc229c72c771a1d5a270c.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "shanghai_souvenir" ---
    shanghai_souvenir_image = visual.ImageStim(
        win=win,
        name='shanghai_souvenir_image', 
        image='images/b56a2466c626aeabd9b381fd66b76cfe.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "xian" ---
    xian_image = visual.ImageStim(
        win=win,
        name='xian_image', 
        image='images/bingmayong.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "xian_souvenir" ---
    xian_souvenir_image = visual.ImageStim(
        win=win,
        name='xian_souvenir_image', 
        image='images/image_with_white_bg.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "hangzhou" ---
    hangzhou_image = visual.ImageStim(
        win=win,
        name='hangzhou_image', 
        image='images/a8650f88a45d59c3a8c8acb1d6665ea8.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "hangzhou_souvenir" ---
    hangzhou_souvenir_image = visual.ImageStim(
        win=win,
        name='hangzhou_souvenir_image', 
        image='images/74c728ba857071a6af0503895941752e.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "shandong" ---
    shangdong_image = visual.ImageStim(
        win=win,
        name='shangdong_image', 
        image='images/cb160d03e8dba34ce6fcf103fcbd2875.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "shangdong_souvenir" ---
    shandong_souvenir_image = visual.ImageStim(
        win=win,
        name='shandong_souvenir_image', 
        image='images/d87dd0f681f4d1a291e2f640f5b557d5.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "tiantan" ---
    tiantan_image = visual.ImageStim(
        win=win,
        name='tiantan_image', 
        image='images/340305d5d74a664595902b651040a220.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "tiantan_souvenir" ---
    tiantan_souvenir_image = visual.ImageStim(
        win=win,
        name='tiantan_souvenir_image', 
        image='images/ff7bf711346910a1fa54c003a366447b.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "xizang" ---
    xizang_image = visual.ImageStim(
        win=win,
        name='xizang_image', 
        image='images/Potala_palace23.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "xizang_souvenir" ---
    xizang_souvenir_image = visual.ImageStim(
        win=win,
        name='xizang_souvenir_image', 
        image='images/5033ef553f6acfcc5efb25106fa20496.jpeg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "great_wall" ---
    great_wall_image = visual.ImageStim(
        win=win,
        name='great_wall_image', 
        image='images/great_wall.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_1" ---
    questionnaire_1_text = visual.TextStim(win=win, name='questionnaire_1_text',
        text='Do you like this tourist attraction?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button',
        depth=-1
    )
    like_button.buttonClock = core.Clock()
    dislike_button = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button',
        depth=-2
    )
    dislike_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    cross_sign_2 = visual.ShapeStim(
        win=win, name='cross_sign_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "great_wall_souvenir" ---
    great_wall_souvenir_image = visual.ImageStim(
        win=win,
        name='great_wall_souvenir_image', 
        image='images/Great Wall product.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "questionnaire_2" ---
    questionnaire_2_text = visual.TextStim(win=win, name='questionnaire_2_text',
        text='Do you like this souvenir?',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    like_button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(-0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='like_button_2',
        depth=-1
    )
    like_button_2.buttonClock = core.Clock()
    dislike_button_2 = visual.ButtonStim(win, 
        text='No', font='Arvo',
        pos=(0.2, -0.2),
        letterHeight=0.02,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[1.0000, 0.9451, 0.7255], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='dislike_button_2',
        depth=-2
    )
    dislike_button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "end" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='Thank you!\n\nThis is the end of the experiment.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_text, welcome_continue],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for welcome_continue
    welcome_continue.keys = []
    welcome_continue.rt = []
    _welcome_continue_allKeys = []
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *welcome_continue* updates
        waitOnFlip = False
        
        # if welcome_continue is starting this frame...
        if welcome_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_continue.frameNStart = frameN  # exact frame index
            welcome_continue.tStart = t  # local t and not account for scr refresh
            welcome_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_continue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_continue.started')
            # update status
            welcome_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_continue.status == STARTED and not waitOnFlip:
            theseKeys = welcome_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_continue_allKeys.extend(theseKeys)
            if len(_welcome_continue_allKeys):
                welcome_continue.keys = _welcome_continue_allKeys[-1].name  # just the last key pressed
                welcome_continue.rt = _welcome_continue_allKeys[-1].rt
                welcome_continue.duration = _welcome_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if welcome_continue.keys in ['', [], None]:  # No response was made
        welcome_continue.keys = None
    thisExp.addData('welcome_continue.keys',welcome_continue.keys)
    if welcome_continue.keys != None:  # we had a response
        thisExp.addData('welcome_continue.rt', welcome_continue.rt)
        thisExp.addData('welcome_continue.duration', welcome_continue.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[intro_text, intro_continue],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_continue
    intro_continue.keys = []
    intro_continue.rt = []
    _intro_continue_allKeys = []
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text.started')
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *intro_continue* updates
        waitOnFlip = False
        
        # if intro_continue is starting this frame...
        if intro_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_continue.frameNStart = frameN  # exact frame index
            intro_continue.tStart = t  # local t and not account for scr refresh
            intro_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_continue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_continue.started')
            # update status
            intro_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_continue.status == STARTED and not waitOnFlip:
            theseKeys = intro_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_continue_allKeys.extend(theseKeys)
            if len(_intro_continue_allKeys):
                intro_continue.keys = _intro_continue_allKeys[-1].name  # just the last key pressed
                intro_continue.rt = _intro_continue_allKeys[-1].rt
                intro_continue.duration = _intro_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    # check responses
    if intro_continue.keys in ['', [], None]:  # No response was made
        intro_continue.keys = None
    thisExp.addData('intro_continue.keys',intro_continue.keys)
    if intro_continue.keys != None:  # we had a response
        thisExp.addData('intro_continue.rt', intro_continue.rt)
        thisExp.addData('intro_continue.duration', intro_continue.duration)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "resting_state_EEG" ---
    # create an object to store info about Routine resting_state_EEG
    resting_state_EEG = data.Routine(
        name='resting_state_EEG',
        components=[cross_sign],
    )
    resting_state_EEG.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for resting_state_EEG
    resting_state_EEG.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    resting_state_EEG.tStart = globalClock.getTime(format='float')
    resting_state_EEG.status = STARTED
    thisExp.addData('resting_state_EEG.started', resting_state_EEG.tStart)
    resting_state_EEG.maxDuration = None
    # keep track of which components have finished
    resting_state_EEGComponents = resting_state_EEG.components
    for thisComponent in resting_state_EEG.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "resting_state_EEG" ---
    resting_state_EEG.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign* updates
        
        # if cross_sign is starting this frame...
        if cross_sign.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign.frameNStart = frameN  # exact frame index
            cross_sign.tStart = t  # local t and not account for scr refresh
            cross_sign.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign.started')
            # update status
            cross_sign.status = STARTED
            cross_sign.setAutoDraw(True)
        
        # if cross_sign is active this frame...
        if cross_sign.status == STARTED:
            # update params
            pass
        
        # if cross_sign is stopping this frame...
        if cross_sign.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 30.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign.tStop = t  # not accounting for scr refresh
                cross_sign.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign.stopped')
                # update status
                cross_sign.status = FINISHED
                cross_sign.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            resting_state_EEG.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resting_state_EEG.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "resting_state_EEG" ---
    for thisComponent in resting_state_EEG.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for resting_state_EEG
    resting_state_EEG.tStop = globalClock.getTime(format='float')
    resting_state_EEG.tStopRefresh = tThisFlipGlobal
    thisExp.addData('resting_state_EEG.stopped', resting_state_EEG.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if resting_state_EEG.maxDurationReached:
        routineTimer.addTime(-resting_state_EEG.maxDuration)
    elif resting_state_EEG.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "resting_state_end" ---
    # create an object to store info about Routine resting_state_end
    resting_state_end = data.Routine(
        name='resting_state_end',
        components=[resting_state_end_text, resting_state_end_continue],
    )
    resting_state_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for resting_state_end_continue
    resting_state_end_continue.keys = []
    resting_state_end_continue.rt = []
    _resting_state_end_continue_allKeys = []
    # store start times for resting_state_end
    resting_state_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    resting_state_end.tStart = globalClock.getTime(format='float')
    resting_state_end.status = STARTED
    thisExp.addData('resting_state_end.started', resting_state_end.tStart)
    resting_state_end.maxDuration = None
    # keep track of which components have finished
    resting_state_endComponents = resting_state_end.components
    for thisComponent in resting_state_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "resting_state_end" ---
    resting_state_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *resting_state_end_text* updates
        
        # if resting_state_end_text is starting this frame...
        if resting_state_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            resting_state_end_text.frameNStart = frameN  # exact frame index
            resting_state_end_text.tStart = t  # local t and not account for scr refresh
            resting_state_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(resting_state_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'resting_state_end_text.started')
            # update status
            resting_state_end_text.status = STARTED
            resting_state_end_text.setAutoDraw(True)
        
        # if resting_state_end_text is active this frame...
        if resting_state_end_text.status == STARTED:
            # update params
            pass
        
        # *resting_state_end_continue* updates
        waitOnFlip = False
        
        # if resting_state_end_continue is starting this frame...
        if resting_state_end_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            resting_state_end_continue.frameNStart = frameN  # exact frame index
            resting_state_end_continue.tStart = t  # local t and not account for scr refresh
            resting_state_end_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(resting_state_end_continue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'resting_state_end_continue.started')
            # update status
            resting_state_end_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(resting_state_end_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(resting_state_end_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if resting_state_end_continue.status == STARTED and not waitOnFlip:
            theseKeys = resting_state_end_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _resting_state_end_continue_allKeys.extend(theseKeys)
            if len(_resting_state_end_continue_allKeys):
                resting_state_end_continue.keys = _resting_state_end_continue_allKeys[-1].name  # just the last key pressed
                resting_state_end_continue.rt = _resting_state_end_continue_allKeys[-1].rt
                resting_state_end_continue.duration = _resting_state_end_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            resting_state_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resting_state_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "resting_state_end" ---
    for thisComponent in resting_state_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for resting_state_end
    resting_state_end.tStop = globalClock.getTime(format='float')
    resting_state_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('resting_state_end.stopped', resting_state_end.tStop)
    # check responses
    if resting_state_end_continue.keys in ['', [], None]:  # No response was made
        resting_state_end_continue.keys = None
    thisExp.addData('resting_state_end_continue.keys',resting_state_end_continue.keys)
    if resting_state_end_continue.keys != None:  # we had a response
        thisExp.addData('resting_state_end_continue.rt', resting_state_end_continue.rt)
        thisExp.addData('resting_state_end_continue.duration', resting_state_end_continue.duration)
    thisExp.nextEntry()
    # the Routine "resting_state_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "huangshan" ---
    # create an object to store info about Routine huangshan
    huangshan = data.Routine(
        name='huangshan',
        components=[huangshan_image],
    )
    huangshan.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for huangshan
    huangshan.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    huangshan.tStart = globalClock.getTime(format='float')
    huangshan.status = STARTED
    thisExp.addData('huangshan.started', huangshan.tStart)
    huangshan.maxDuration = None
    # keep track of which components have finished
    huangshanComponents = huangshan.components
    for thisComponent in huangshan.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "huangshan" ---
    huangshan.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *huangshan_image* updates
        
        # if huangshan_image is starting this frame...
        if huangshan_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            huangshan_image.frameNStart = frameN  # exact frame index
            huangshan_image.tStart = t  # local t and not account for scr refresh
            huangshan_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(huangshan_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'huangshan_image.started')
            # update status
            huangshan_image.status = STARTED
            huangshan_image.setAutoDraw(True)
        
        # if huangshan_image is active this frame...
        if huangshan_image.status == STARTED:
            # update params
            pass
        
        # if huangshan_image is stopping this frame...
        if huangshan_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                huangshan_image.tStop = t  # not accounting for scr refresh
                huangshan_image.tStopRefresh = tThisFlipGlobal  # on global time
                huangshan_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'huangshan_image.stopped')
                # update status
                huangshan_image.status = FINISHED
                huangshan_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            huangshan.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in huangshan.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "huangshan" ---
    for thisComponent in huangshan.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for huangshan
    huangshan.tStop = globalClock.getTime(format='float')
    huangshan.tStopRefresh = tThisFlipGlobal
    thisExp.addData('huangshan.stopped', huangshan.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if huangshan.maxDurationReached:
        routineTimer.addTime(-huangshan.maxDuration)
    elif huangshan.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "huangshan_souvenir" ---
    # create an object to store info about Routine huangshan_souvenir
    huangshan_souvenir = data.Routine(
        name='huangshan_souvenir',
        components=[huangshan_souvenir_image],
    )
    huangshan_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for huangshan_souvenir
    huangshan_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    huangshan_souvenir.tStart = globalClock.getTime(format='float')
    huangshan_souvenir.status = STARTED
    thisExp.addData('huangshan_souvenir.started', huangshan_souvenir.tStart)
    huangshan_souvenir.maxDuration = None
    # keep track of which components have finished
    huangshan_souvenirComponents = huangshan_souvenir.components
    for thisComponent in huangshan_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "huangshan_souvenir" ---
    huangshan_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *huangshan_souvenir_image* updates
        
        # if huangshan_souvenir_image is starting this frame...
        if huangshan_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            huangshan_souvenir_image.frameNStart = frameN  # exact frame index
            huangshan_souvenir_image.tStart = t  # local t and not account for scr refresh
            huangshan_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(huangshan_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'huangshan_souvenir_image.started')
            # update status
            huangshan_souvenir_image.status = STARTED
            huangshan_souvenir_image.setAutoDraw(True)
        
        # if huangshan_souvenir_image is active this frame...
        if huangshan_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if huangshan_souvenir_image is stopping this frame...
        if huangshan_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                huangshan_souvenir_image.tStop = t  # not accounting for scr refresh
                huangshan_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                huangshan_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'huangshan_souvenir_image.stopped')
                # update status
                huangshan_souvenir_image.status = FINISHED
                huangshan_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            huangshan_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in huangshan_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "huangshan_souvenir" ---
    for thisComponent in huangshan_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for huangshan_souvenir
    huangshan_souvenir.tStop = globalClock.getTime(format='float')
    huangshan_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('huangshan_souvenir.stopped', huangshan_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if huangshan_souvenir.maxDurationReached:
        routineTimer.addTime(-huangshan_souvenir.maxDuration)
    elif huangshan_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "forbidden_city" ---
    # create an object to store info about Routine forbidden_city
    forbidden_city = data.Routine(
        name='forbidden_city',
        components=[forbidden_city_image],
    )
    forbidden_city.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for forbidden_city
    forbidden_city.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    forbidden_city.tStart = globalClock.getTime(format='float')
    forbidden_city.status = STARTED
    thisExp.addData('forbidden_city.started', forbidden_city.tStart)
    forbidden_city.maxDuration = None
    # keep track of which components have finished
    forbidden_cityComponents = forbidden_city.components
    for thisComponent in forbidden_city.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "forbidden_city" ---
    forbidden_city.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *forbidden_city_image* updates
        
        # if forbidden_city_image is starting this frame...
        if forbidden_city_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            forbidden_city_image.frameNStart = frameN  # exact frame index
            forbidden_city_image.tStart = t  # local t and not account for scr refresh
            forbidden_city_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(forbidden_city_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'forbidden_city_image.started')
            # update status
            forbidden_city_image.status = STARTED
            forbidden_city_image.setAutoDraw(True)
        
        # if forbidden_city_image is active this frame...
        if forbidden_city_image.status == STARTED:
            # update params
            pass
        
        # if forbidden_city_image is stopping this frame...
        if forbidden_city_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                forbidden_city_image.tStop = t  # not accounting for scr refresh
                forbidden_city_image.tStopRefresh = tThisFlipGlobal  # on global time
                forbidden_city_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'forbidden_city_image.stopped')
                # update status
                forbidden_city_image.status = FINISHED
                forbidden_city_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            forbidden_city.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in forbidden_city.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "forbidden_city" ---
    for thisComponent in forbidden_city.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for forbidden_city
    forbidden_city.tStop = globalClock.getTime(format='float')
    forbidden_city.tStopRefresh = tThisFlipGlobal
    thisExp.addData('forbidden_city.stopped', forbidden_city.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if forbidden_city.maxDurationReached:
        routineTimer.addTime(-forbidden_city.maxDuration)
    elif forbidden_city.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "forbidden_city_souvenir" ---
    # create an object to store info about Routine forbidden_city_souvenir
    forbidden_city_souvenir = data.Routine(
        name='forbidden_city_souvenir',
        components=[forbidden_city_souvenir_image],
    )
    forbidden_city_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for forbidden_city_souvenir
    forbidden_city_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    forbidden_city_souvenir.tStart = globalClock.getTime(format='float')
    forbidden_city_souvenir.status = STARTED
    thisExp.addData('forbidden_city_souvenir.started', forbidden_city_souvenir.tStart)
    forbidden_city_souvenir.maxDuration = None
    # keep track of which components have finished
    forbidden_city_souvenirComponents = forbidden_city_souvenir.components
    for thisComponent in forbidden_city_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "forbidden_city_souvenir" ---
    forbidden_city_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *forbidden_city_souvenir_image* updates
        
        # if forbidden_city_souvenir_image is starting this frame...
        if forbidden_city_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            forbidden_city_souvenir_image.frameNStart = frameN  # exact frame index
            forbidden_city_souvenir_image.tStart = t  # local t and not account for scr refresh
            forbidden_city_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(forbidden_city_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'forbidden_city_souvenir_image.started')
            # update status
            forbidden_city_souvenir_image.status = STARTED
            forbidden_city_souvenir_image.setAutoDraw(True)
        
        # if forbidden_city_souvenir_image is active this frame...
        if forbidden_city_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if forbidden_city_souvenir_image is stopping this frame...
        if forbidden_city_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                forbidden_city_souvenir_image.tStop = t  # not accounting for scr refresh
                forbidden_city_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                forbidden_city_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'forbidden_city_souvenir_image.stopped')
                # update status
                forbidden_city_souvenir_image.status = FINISHED
                forbidden_city_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            forbidden_city_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in forbidden_city_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "forbidden_city_souvenir" ---
    for thisComponent in forbidden_city_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for forbidden_city_souvenir
    forbidden_city_souvenir.tStop = globalClock.getTime(format='float')
    forbidden_city_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('forbidden_city_souvenir.stopped', forbidden_city_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if forbidden_city_souvenir.maxDurationReached:
        routineTimer.addTime(-forbidden_city_souvenir.maxDuration)
    elif forbidden_city_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "suzhou" ---
    # create an object to store info about Routine suzhou
    suzhou = data.Routine(
        name='suzhou',
        components=[suzhou_image],
    )
    suzhou.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for suzhou
    suzhou.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    suzhou.tStart = globalClock.getTime(format='float')
    suzhou.status = STARTED
    thisExp.addData('suzhou.started', suzhou.tStart)
    suzhou.maxDuration = None
    # keep track of which components have finished
    suzhouComponents = suzhou.components
    for thisComponent in suzhou.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "suzhou" ---
    suzhou.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *suzhou_image* updates
        
        # if suzhou_image is starting this frame...
        if suzhou_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            suzhou_image.frameNStart = frameN  # exact frame index
            suzhou_image.tStart = t  # local t and not account for scr refresh
            suzhou_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(suzhou_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'suzhou_image.started')
            # update status
            suzhou_image.status = STARTED
            suzhou_image.setAutoDraw(True)
        
        # if suzhou_image is active this frame...
        if suzhou_image.status == STARTED:
            # update params
            pass
        
        # if suzhou_image is stopping this frame...
        if suzhou_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                suzhou_image.tStop = t  # not accounting for scr refresh
                suzhou_image.tStopRefresh = tThisFlipGlobal  # on global time
                suzhou_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'suzhou_image.stopped')
                # update status
                suzhou_image.status = FINISHED
                suzhou_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            suzhou.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in suzhou.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "suzhou" ---
    for thisComponent in suzhou.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for suzhou
    suzhou.tStop = globalClock.getTime(format='float')
    suzhou.tStopRefresh = tThisFlipGlobal
    thisExp.addData('suzhou.stopped', suzhou.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if suzhou.maxDurationReached:
        routineTimer.addTime(-suzhou.maxDuration)
    elif suzhou.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "suzhou_souvenir" ---
    # create an object to store info about Routine suzhou_souvenir
    suzhou_souvenir = data.Routine(
        name='suzhou_souvenir',
        components=[suzhou_souvenir_image],
    )
    suzhou_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for suzhou_souvenir
    suzhou_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    suzhou_souvenir.tStart = globalClock.getTime(format='float')
    suzhou_souvenir.status = STARTED
    thisExp.addData('suzhou_souvenir.started', suzhou_souvenir.tStart)
    suzhou_souvenir.maxDuration = None
    # keep track of which components have finished
    suzhou_souvenirComponents = suzhou_souvenir.components
    for thisComponent in suzhou_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "suzhou_souvenir" ---
    suzhou_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *suzhou_souvenir_image* updates
        
        # if suzhou_souvenir_image is starting this frame...
        if suzhou_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            suzhou_souvenir_image.frameNStart = frameN  # exact frame index
            suzhou_souvenir_image.tStart = t  # local t and not account for scr refresh
            suzhou_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(suzhou_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'suzhou_souvenir_image.started')
            # update status
            suzhou_souvenir_image.status = STARTED
            suzhou_souvenir_image.setAutoDraw(True)
        
        # if suzhou_souvenir_image is active this frame...
        if suzhou_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if suzhou_souvenir_image is stopping this frame...
        if suzhou_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                suzhou_souvenir_image.tStop = t  # not accounting for scr refresh
                suzhou_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                suzhou_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'suzhou_souvenir_image.stopped')
                # update status
                suzhou_souvenir_image.status = FINISHED
                suzhou_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            suzhou_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in suzhou_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "suzhou_souvenir" ---
    for thisComponent in suzhou_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for suzhou_souvenir
    suzhou_souvenir.tStop = globalClock.getTime(format='float')
    suzhou_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('suzhou_souvenir.stopped', suzhou_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if suzhou_souvenir.maxDurationReached:
        routineTimer.addTime(-suzhou_souvenir.maxDuration)
    elif suzhou_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "shanghai" ---
    # create an object to store info about Routine shanghai
    shanghai = data.Routine(
        name='shanghai',
        components=[shanghai_image],
    )
    shanghai.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shanghai
    shanghai.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shanghai.tStart = globalClock.getTime(format='float')
    shanghai.status = STARTED
    thisExp.addData('shanghai.started', shanghai.tStart)
    shanghai.maxDuration = None
    # keep track of which components have finished
    shanghaiComponents = shanghai.components
    for thisComponent in shanghai.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shanghai" ---
    shanghai.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *shanghai_image* updates
        
        # if shanghai_image is starting this frame...
        if shanghai_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            shanghai_image.frameNStart = frameN  # exact frame index
            shanghai_image.tStart = t  # local t and not account for scr refresh
            shanghai_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shanghai_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shanghai_image.started')
            # update status
            shanghai_image.status = STARTED
            shanghai_image.setAutoDraw(True)
        
        # if shanghai_image is active this frame...
        if shanghai_image.status == STARTED:
            # update params
            pass
        
        # if shanghai_image is stopping this frame...
        if shanghai_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                shanghai_image.tStop = t  # not accounting for scr refresh
                shanghai_image.tStopRefresh = tThisFlipGlobal  # on global time
                shanghai_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shanghai_image.stopped')
                # update status
                shanghai_image.status = FINISHED
                shanghai_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shanghai.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shanghai.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shanghai" ---
    for thisComponent in shanghai.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shanghai
    shanghai.tStop = globalClock.getTime(format='float')
    shanghai.tStopRefresh = tThisFlipGlobal
    thisExp.addData('shanghai.stopped', shanghai.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if shanghai.maxDurationReached:
        routineTimer.addTime(-shanghai.maxDuration)
    elif shanghai.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "shanghai_souvenir" ---
    # create an object to store info about Routine shanghai_souvenir
    shanghai_souvenir = data.Routine(
        name='shanghai_souvenir',
        components=[shanghai_souvenir_image],
    )
    shanghai_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shanghai_souvenir
    shanghai_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shanghai_souvenir.tStart = globalClock.getTime(format='float')
    shanghai_souvenir.status = STARTED
    thisExp.addData('shanghai_souvenir.started', shanghai_souvenir.tStart)
    shanghai_souvenir.maxDuration = None
    # keep track of which components have finished
    shanghai_souvenirComponents = shanghai_souvenir.components
    for thisComponent in shanghai_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shanghai_souvenir" ---
    shanghai_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *shanghai_souvenir_image* updates
        
        # if shanghai_souvenir_image is starting this frame...
        if shanghai_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            shanghai_souvenir_image.frameNStart = frameN  # exact frame index
            shanghai_souvenir_image.tStart = t  # local t and not account for scr refresh
            shanghai_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shanghai_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shanghai_souvenir_image.started')
            # update status
            shanghai_souvenir_image.status = STARTED
            shanghai_souvenir_image.setAutoDraw(True)
        
        # if shanghai_souvenir_image is active this frame...
        if shanghai_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if shanghai_souvenir_image is stopping this frame...
        if shanghai_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                shanghai_souvenir_image.tStop = t  # not accounting for scr refresh
                shanghai_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                shanghai_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shanghai_souvenir_image.stopped')
                # update status
                shanghai_souvenir_image.status = FINISHED
                shanghai_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shanghai_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shanghai_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shanghai_souvenir" ---
    for thisComponent in shanghai_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shanghai_souvenir
    shanghai_souvenir.tStop = globalClock.getTime(format='float')
    shanghai_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('shanghai_souvenir.stopped', shanghai_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if shanghai_souvenir.maxDurationReached:
        routineTimer.addTime(-shanghai_souvenir.maxDuration)
    elif shanghai_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "xian" ---
    # create an object to store info about Routine xian
    xian = data.Routine(
        name='xian',
        components=[xian_image],
    )
    xian.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for xian
    xian.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    xian.tStart = globalClock.getTime(format='float')
    xian.status = STARTED
    thisExp.addData('xian.started', xian.tStart)
    xian.maxDuration = None
    # keep track of which components have finished
    xianComponents = xian.components
    for thisComponent in xian.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "xian" ---
    xian.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *xian_image* updates
        
        # if xian_image is starting this frame...
        if xian_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xian_image.frameNStart = frameN  # exact frame index
            xian_image.tStart = t  # local t and not account for scr refresh
            xian_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xian_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xian_image.started')
            # update status
            xian_image.status = STARTED
            xian_image.setAutoDraw(True)
        
        # if xian_image is active this frame...
        if xian_image.status == STARTED:
            # update params
            pass
        
        # if xian_image is stopping this frame...
        if xian_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                xian_image.tStop = t  # not accounting for scr refresh
                xian_image.tStopRefresh = tThisFlipGlobal  # on global time
                xian_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xian_image.stopped')
                # update status
                xian_image.status = FINISHED
                xian_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            xian.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in xian.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "xian" ---
    for thisComponent in xian.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for xian
    xian.tStop = globalClock.getTime(format='float')
    xian.tStopRefresh = tThisFlipGlobal
    thisExp.addData('xian.stopped', xian.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if xian.maxDurationReached:
        routineTimer.addTime(-xian.maxDuration)
    elif xian.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "xian_souvenir" ---
    # create an object to store info about Routine xian_souvenir
    xian_souvenir = data.Routine(
        name='xian_souvenir',
        components=[xian_souvenir_image],
    )
    xian_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for xian_souvenir
    xian_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    xian_souvenir.tStart = globalClock.getTime(format='float')
    xian_souvenir.status = STARTED
    thisExp.addData('xian_souvenir.started', xian_souvenir.tStart)
    xian_souvenir.maxDuration = None
    # keep track of which components have finished
    xian_souvenirComponents = xian_souvenir.components
    for thisComponent in xian_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "xian_souvenir" ---
    xian_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *xian_souvenir_image* updates
        
        # if xian_souvenir_image is starting this frame...
        if xian_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xian_souvenir_image.frameNStart = frameN  # exact frame index
            xian_souvenir_image.tStart = t  # local t and not account for scr refresh
            xian_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xian_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xian_souvenir_image.started')
            # update status
            xian_souvenir_image.status = STARTED
            xian_souvenir_image.setAutoDraw(True)
        
        # if xian_souvenir_image is active this frame...
        if xian_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if xian_souvenir_image is stopping this frame...
        if xian_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                xian_souvenir_image.tStop = t  # not accounting for scr refresh
                xian_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                xian_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xian_souvenir_image.stopped')
                # update status
                xian_souvenir_image.status = FINISHED
                xian_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            xian_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in xian_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "xian_souvenir" ---
    for thisComponent in xian_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for xian_souvenir
    xian_souvenir.tStop = globalClock.getTime(format='float')
    xian_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('xian_souvenir.stopped', xian_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if xian_souvenir.maxDurationReached:
        routineTimer.addTime(-xian_souvenir.maxDuration)
    elif xian_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "hangzhou" ---
    # create an object to store info about Routine hangzhou
    hangzhou = data.Routine(
        name='hangzhou',
        components=[hangzhou_image],
    )
    hangzhou.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for hangzhou
    hangzhou.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    hangzhou.tStart = globalClock.getTime(format='float')
    hangzhou.status = STARTED
    thisExp.addData('hangzhou.started', hangzhou.tStart)
    hangzhou.maxDuration = None
    # keep track of which components have finished
    hangzhouComponents = hangzhou.components
    for thisComponent in hangzhou.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "hangzhou" ---
    hangzhou.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *hangzhou_image* updates
        
        # if hangzhou_image is starting this frame...
        if hangzhou_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hangzhou_image.frameNStart = frameN  # exact frame index
            hangzhou_image.tStart = t  # local t and not account for scr refresh
            hangzhou_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hangzhou_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'hangzhou_image.started')
            # update status
            hangzhou_image.status = STARTED
            hangzhou_image.setAutoDraw(True)
        
        # if hangzhou_image is active this frame...
        if hangzhou_image.status == STARTED:
            # update params
            pass
        
        # if hangzhou_image is stopping this frame...
        if hangzhou_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                hangzhou_image.tStop = t  # not accounting for scr refresh
                hangzhou_image.tStopRefresh = tThisFlipGlobal  # on global time
                hangzhou_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'hangzhou_image.stopped')
                # update status
                hangzhou_image.status = FINISHED
                hangzhou_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            hangzhou.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in hangzhou.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hangzhou" ---
    for thisComponent in hangzhou.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for hangzhou
    hangzhou.tStop = globalClock.getTime(format='float')
    hangzhou.tStopRefresh = tThisFlipGlobal
    thisExp.addData('hangzhou.stopped', hangzhou.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if hangzhou.maxDurationReached:
        routineTimer.addTime(-hangzhou.maxDuration)
    elif hangzhou.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "hangzhou_souvenir" ---
    # create an object to store info about Routine hangzhou_souvenir
    hangzhou_souvenir = data.Routine(
        name='hangzhou_souvenir',
        components=[hangzhou_souvenir_image],
    )
    hangzhou_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for hangzhou_souvenir
    hangzhou_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    hangzhou_souvenir.tStart = globalClock.getTime(format='float')
    hangzhou_souvenir.status = STARTED
    thisExp.addData('hangzhou_souvenir.started', hangzhou_souvenir.tStart)
    hangzhou_souvenir.maxDuration = None
    # keep track of which components have finished
    hangzhou_souvenirComponents = hangzhou_souvenir.components
    for thisComponent in hangzhou_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "hangzhou_souvenir" ---
    hangzhou_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *hangzhou_souvenir_image* updates
        
        # if hangzhou_souvenir_image is starting this frame...
        if hangzhou_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hangzhou_souvenir_image.frameNStart = frameN  # exact frame index
            hangzhou_souvenir_image.tStart = t  # local t and not account for scr refresh
            hangzhou_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hangzhou_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'hangzhou_souvenir_image.started')
            # update status
            hangzhou_souvenir_image.status = STARTED
            hangzhou_souvenir_image.setAutoDraw(True)
        
        # if hangzhou_souvenir_image is active this frame...
        if hangzhou_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if hangzhou_souvenir_image is stopping this frame...
        if hangzhou_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                hangzhou_souvenir_image.tStop = t  # not accounting for scr refresh
                hangzhou_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                hangzhou_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'hangzhou_souvenir_image.stopped')
                # update status
                hangzhou_souvenir_image.status = FINISHED
                hangzhou_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            hangzhou_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in hangzhou_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hangzhou_souvenir" ---
    for thisComponent in hangzhou_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for hangzhou_souvenir
    hangzhou_souvenir.tStop = globalClock.getTime(format='float')
    hangzhou_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('hangzhou_souvenir.stopped', hangzhou_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if hangzhou_souvenir.maxDurationReached:
        routineTimer.addTime(-hangzhou_souvenir.maxDuration)
    elif hangzhou_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "shandong" ---
    # create an object to store info about Routine shandong
    shandong = data.Routine(
        name='shandong',
        components=[shangdong_image],
    )
    shandong.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shandong
    shandong.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shandong.tStart = globalClock.getTime(format='float')
    shandong.status = STARTED
    thisExp.addData('shandong.started', shandong.tStart)
    shandong.maxDuration = None
    # keep track of which components have finished
    shandongComponents = shandong.components
    for thisComponent in shandong.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shandong" ---
    shandong.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *shangdong_image* updates
        
        # if shangdong_image is starting this frame...
        if shangdong_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            shangdong_image.frameNStart = frameN  # exact frame index
            shangdong_image.tStart = t  # local t and not account for scr refresh
            shangdong_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shangdong_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shangdong_image.started')
            # update status
            shangdong_image.status = STARTED
            shangdong_image.setAutoDraw(True)
        
        # if shangdong_image is active this frame...
        if shangdong_image.status == STARTED:
            # update params
            pass
        
        # if shangdong_image is stopping this frame...
        if shangdong_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                shangdong_image.tStop = t  # not accounting for scr refresh
                shangdong_image.tStopRefresh = tThisFlipGlobal  # on global time
                shangdong_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shangdong_image.stopped')
                # update status
                shangdong_image.status = FINISHED
                shangdong_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shandong.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shandong.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shandong" ---
    for thisComponent in shandong.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shandong
    shandong.tStop = globalClock.getTime(format='float')
    shandong.tStopRefresh = tThisFlipGlobal
    thisExp.addData('shandong.stopped', shandong.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if shandong.maxDurationReached:
        routineTimer.addTime(-shandong.maxDuration)
    elif shandong.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "shangdong_souvenir" ---
    # create an object to store info about Routine shangdong_souvenir
    shangdong_souvenir = data.Routine(
        name='shangdong_souvenir',
        components=[shandong_souvenir_image],
    )
    shangdong_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shangdong_souvenir
    shangdong_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shangdong_souvenir.tStart = globalClock.getTime(format='float')
    shangdong_souvenir.status = STARTED
    thisExp.addData('shangdong_souvenir.started', shangdong_souvenir.tStart)
    shangdong_souvenir.maxDuration = None
    # keep track of which components have finished
    shangdong_souvenirComponents = shangdong_souvenir.components
    for thisComponent in shangdong_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shangdong_souvenir" ---
    shangdong_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *shandong_souvenir_image* updates
        
        # if shandong_souvenir_image is starting this frame...
        if shandong_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            shandong_souvenir_image.frameNStart = frameN  # exact frame index
            shandong_souvenir_image.tStart = t  # local t and not account for scr refresh
            shandong_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shandong_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shandong_souvenir_image.started')
            # update status
            shandong_souvenir_image.status = STARTED
            shandong_souvenir_image.setAutoDraw(True)
        
        # if shandong_souvenir_image is active this frame...
        if shandong_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if shandong_souvenir_image is stopping this frame...
        if shandong_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                shandong_souvenir_image.tStop = t  # not accounting for scr refresh
                shandong_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                shandong_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shandong_souvenir_image.stopped')
                # update status
                shandong_souvenir_image.status = FINISHED
                shandong_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shangdong_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shangdong_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shangdong_souvenir" ---
    for thisComponent in shangdong_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shangdong_souvenir
    shangdong_souvenir.tStop = globalClock.getTime(format='float')
    shangdong_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('shangdong_souvenir.stopped', shangdong_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if shangdong_souvenir.maxDurationReached:
        routineTimer.addTime(-shangdong_souvenir.maxDuration)
    elif shangdong_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "tiantan" ---
    # create an object to store info about Routine tiantan
    tiantan = data.Routine(
        name='tiantan',
        components=[tiantan_image],
    )
    tiantan.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for tiantan
    tiantan.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    tiantan.tStart = globalClock.getTime(format='float')
    tiantan.status = STARTED
    thisExp.addData('tiantan.started', tiantan.tStart)
    tiantan.maxDuration = None
    # keep track of which components have finished
    tiantanComponents = tiantan.components
    for thisComponent in tiantan.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "tiantan" ---
    tiantan.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *tiantan_image* updates
        
        # if tiantan_image is starting this frame...
        if tiantan_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tiantan_image.frameNStart = frameN  # exact frame index
            tiantan_image.tStart = t  # local t and not account for scr refresh
            tiantan_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tiantan_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tiantan_image.started')
            # update status
            tiantan_image.status = STARTED
            tiantan_image.setAutoDraw(True)
        
        # if tiantan_image is active this frame...
        if tiantan_image.status == STARTED:
            # update params
            pass
        
        # if tiantan_image is stopping this frame...
        if tiantan_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                tiantan_image.tStop = t  # not accounting for scr refresh
                tiantan_image.tStopRefresh = tThisFlipGlobal  # on global time
                tiantan_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tiantan_image.stopped')
                # update status
                tiantan_image.status = FINISHED
                tiantan_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            tiantan.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in tiantan.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "tiantan" ---
    for thisComponent in tiantan.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for tiantan
    tiantan.tStop = globalClock.getTime(format='float')
    tiantan.tStopRefresh = tThisFlipGlobal
    thisExp.addData('tiantan.stopped', tiantan.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if tiantan.maxDurationReached:
        routineTimer.addTime(-tiantan.maxDuration)
    elif tiantan.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "tiantan_souvenir" ---
    # create an object to store info about Routine tiantan_souvenir
    tiantan_souvenir = data.Routine(
        name='tiantan_souvenir',
        components=[tiantan_souvenir_image],
    )
    tiantan_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for tiantan_souvenir
    tiantan_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    tiantan_souvenir.tStart = globalClock.getTime(format='float')
    tiantan_souvenir.status = STARTED
    thisExp.addData('tiantan_souvenir.started', tiantan_souvenir.tStart)
    tiantan_souvenir.maxDuration = None
    # keep track of which components have finished
    tiantan_souvenirComponents = tiantan_souvenir.components
    for thisComponent in tiantan_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "tiantan_souvenir" ---
    tiantan_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *tiantan_souvenir_image* updates
        
        # if tiantan_souvenir_image is starting this frame...
        if tiantan_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tiantan_souvenir_image.frameNStart = frameN  # exact frame index
            tiantan_souvenir_image.tStart = t  # local t and not account for scr refresh
            tiantan_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tiantan_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tiantan_souvenir_image.started')
            # update status
            tiantan_souvenir_image.status = STARTED
            tiantan_souvenir_image.setAutoDraw(True)
        
        # if tiantan_souvenir_image is active this frame...
        if tiantan_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if tiantan_souvenir_image is stopping this frame...
        if tiantan_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                tiantan_souvenir_image.tStop = t  # not accounting for scr refresh
                tiantan_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                tiantan_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tiantan_souvenir_image.stopped')
                # update status
                tiantan_souvenir_image.status = FINISHED
                tiantan_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            tiantan_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in tiantan_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "tiantan_souvenir" ---
    for thisComponent in tiantan_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for tiantan_souvenir
    tiantan_souvenir.tStop = globalClock.getTime(format='float')
    tiantan_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('tiantan_souvenir.stopped', tiantan_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if tiantan_souvenir.maxDurationReached:
        routineTimer.addTime(-tiantan_souvenir.maxDuration)
    elif tiantan_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "xizang" ---
    # create an object to store info about Routine xizang
    xizang = data.Routine(
        name='xizang',
        components=[xizang_image],
    )
    xizang.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for xizang
    xizang.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    xizang.tStart = globalClock.getTime(format='float')
    xizang.status = STARTED
    thisExp.addData('xizang.started', xizang.tStart)
    xizang.maxDuration = None
    # keep track of which components have finished
    xizangComponents = xizang.components
    for thisComponent in xizang.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "xizang" ---
    xizang.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *xizang_image* updates
        
        # if xizang_image is starting this frame...
        if xizang_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xizang_image.frameNStart = frameN  # exact frame index
            xizang_image.tStart = t  # local t and not account for scr refresh
            xizang_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xizang_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xizang_image.started')
            # update status
            xizang_image.status = STARTED
            xizang_image.setAutoDraw(True)
        
        # if xizang_image is active this frame...
        if xizang_image.status == STARTED:
            # update params
            pass
        
        # if xizang_image is stopping this frame...
        if xizang_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                xizang_image.tStop = t  # not accounting for scr refresh
                xizang_image.tStopRefresh = tThisFlipGlobal  # on global time
                xizang_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xizang_image.stopped')
                # update status
                xizang_image.status = FINISHED
                xizang_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            xizang.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in xizang.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "xizang" ---
    for thisComponent in xizang.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for xizang
    xizang.tStop = globalClock.getTime(format='float')
    xizang.tStopRefresh = tThisFlipGlobal
    thisExp.addData('xizang.stopped', xizang.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if xizang.maxDurationReached:
        routineTimer.addTime(-xizang.maxDuration)
    elif xizang.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "xizang_souvenir" ---
    # create an object to store info about Routine xizang_souvenir
    xizang_souvenir = data.Routine(
        name='xizang_souvenir',
        components=[xizang_souvenir_image],
    )
    xizang_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for xizang_souvenir
    xizang_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    xizang_souvenir.tStart = globalClock.getTime(format='float')
    xizang_souvenir.status = STARTED
    thisExp.addData('xizang_souvenir.started', xizang_souvenir.tStart)
    xizang_souvenir.maxDuration = None
    # keep track of which components have finished
    xizang_souvenirComponents = xizang_souvenir.components
    for thisComponent in xizang_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "xizang_souvenir" ---
    xizang_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *xizang_souvenir_image* updates
        
        # if xizang_souvenir_image is starting this frame...
        if xizang_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xizang_souvenir_image.frameNStart = frameN  # exact frame index
            xizang_souvenir_image.tStart = t  # local t and not account for scr refresh
            xizang_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xizang_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xizang_souvenir_image.started')
            # update status
            xizang_souvenir_image.status = STARTED
            xizang_souvenir_image.setAutoDraw(True)
        
        # if xizang_souvenir_image is active this frame...
        if xizang_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if xizang_souvenir_image is stopping this frame...
        if xizang_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                xizang_souvenir_image.tStop = t  # not accounting for scr refresh
                xizang_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                xizang_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xizang_souvenir_image.stopped')
                # update status
                xizang_souvenir_image.status = FINISHED
                xizang_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            xizang_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in xizang_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "xizang_souvenir" ---
    for thisComponent in xizang_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for xizang_souvenir
    xizang_souvenir.tStop = globalClock.getTime(format='float')
    xizang_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('xizang_souvenir.stopped', xizang_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if xizang_souvenir.maxDurationReached:
        routineTimer.addTime(-xizang_souvenir.maxDuration)
    elif xizang_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "great_wall" ---
    # create an object to store info about Routine great_wall
    great_wall = data.Routine(
        name='great_wall',
        components=[great_wall_image],
    )
    great_wall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for great_wall
    great_wall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    great_wall.tStart = globalClock.getTime(format='float')
    great_wall.status = STARTED
    thisExp.addData('great_wall.started', great_wall.tStart)
    great_wall.maxDuration = None
    # keep track of which components have finished
    great_wallComponents = great_wall.components
    for thisComponent in great_wall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "great_wall" ---
    great_wall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *great_wall_image* updates
        
        # if great_wall_image is starting this frame...
        if great_wall_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            great_wall_image.frameNStart = frameN  # exact frame index
            great_wall_image.tStart = t  # local t and not account for scr refresh
            great_wall_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(great_wall_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'great_wall_image.started')
            # update status
            great_wall_image.status = STARTED
            great_wall_image.setAutoDraw(True)
        
        # if great_wall_image is active this frame...
        if great_wall_image.status == STARTED:
            # update params
            pass
        
        # if great_wall_image is stopping this frame...
        if great_wall_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                great_wall_image.tStop = t  # not accounting for scr refresh
                great_wall_image.tStopRefresh = tThisFlipGlobal  # on global time
                great_wall_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'great_wall_image.stopped')
                # update status
                great_wall_image.status = FINISHED
                great_wall_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            great_wall.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in great_wall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "great_wall" ---
    for thisComponent in great_wall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for great_wall
    great_wall.tStop = globalClock.getTime(format='float')
    great_wall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('great_wall.stopped', great_wall.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if great_wall.maxDurationReached:
        routineTimer.addTime(-great_wall.maxDuration)
    elif great_wall.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_1" ---
    # create an object to store info about Routine questionnaire_1
    questionnaire_1 = data.Routine(
        name='questionnaire_1',
        components=[questionnaire_1_text, like_button, dislike_button],
    )
    questionnaire_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button to account for continued clicks & clear times on/off
    like_button.reset()
    # reset dislike_button to account for continued clicks & clear times on/off
    dislike_button.reset()
    # store start times for questionnaire_1
    questionnaire_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_1.tStart = globalClock.getTime(format='float')
    questionnaire_1.status = STARTED
    thisExp.addData('questionnaire_1.started', questionnaire_1.tStart)
    questionnaire_1.maxDuration = None
    # keep track of which components have finished
    questionnaire_1Components = questionnaire_1.components
    for thisComponent in questionnaire_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_1" ---
    questionnaire_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_1_text* updates
        
        # if questionnaire_1_text is starting this frame...
        if questionnaire_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_1_text.frameNStart = frameN  # exact frame index
            questionnaire_1_text.tStart = t  # local t and not account for scr refresh
            questionnaire_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_1_text.started')
            # update status
            questionnaire_1_text.status = STARTED
            questionnaire_1_text.setAutoDraw(True)
        
        # if questionnaire_1_text is active this frame...
        if questionnaire_1_text.status == STARTED:
            # update params
            pass
        # *like_button* updates
        
        # if like_button is starting this frame...
        if like_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button.frameNStart = frameN  # exact frame index
            like_button.tStart = t  # local t and not account for scr refresh
            like_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button.started')
            # update status
            like_button.status = STARTED
            win.callOnFlip(like_button.buttonClock.reset)
            like_button.setAutoDraw(True)
        
        # if like_button is active this frame...
        if like_button.status == STARTED:
            # update params
            pass
            # check whether like_button has been pressed
            if like_button.isClicked:
                if not like_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button.timesOn.append(like_button.buttonClock.getTime())
                    like_button.timesOff.append(like_button.buttonClock.getTime())
                elif len(like_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button.timesOff[-1] = like_button.buttonClock.getTime()
                if not like_button.wasClicked:
                    # end routine when like_button is clicked
                    continueRoutine = False
                if not like_button.wasClicked:
                    # run callback code when like_button is clicked
                    pass
        # take note of whether like_button was clicked, so that next frame we know if clicks are new
        like_button.wasClicked = like_button.isClicked and like_button.status == STARTED
        # *dislike_button* updates
        
        # if dislike_button is starting this frame...
        if dislike_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button.frameNStart = frameN  # exact frame index
            dislike_button.tStart = t  # local t and not account for scr refresh
            dislike_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button.started')
            # update status
            dislike_button.status = STARTED
            win.callOnFlip(dislike_button.buttonClock.reset)
            dislike_button.setAutoDraw(True)
        
        # if dislike_button is active this frame...
        if dislike_button.status == STARTED:
            # update params
            pass
            # check whether dislike_button has been pressed
            if dislike_button.isClicked:
                if not dislike_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button.timesOn.append(dislike_button.buttonClock.getTime())
                    dislike_button.timesOff.append(dislike_button.buttonClock.getTime())
                elif len(dislike_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button.timesOff[-1] = dislike_button.buttonClock.getTime()
                if not dislike_button.wasClicked:
                    # end routine when dislike_button is clicked
                    continueRoutine = False
                if not dislike_button.wasClicked:
                    # run callback code when dislike_button is clicked
                    pass
        # take note of whether dislike_button was clicked, so that next frame we know if clicks are new
        dislike_button.wasClicked = dislike_button.isClicked and dislike_button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_1" ---
    for thisComponent in questionnaire_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_1
    questionnaire_1.tStop = globalClock.getTime(format='float')
    questionnaire_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_1.stopped', questionnaire_1.tStop)
    thisExp.addData('like_button.numClicks', like_button.numClicks)
    if like_button.numClicks:
       thisExp.addData('like_button.timesOn', like_button.timesOn)
       thisExp.addData('like_button.timesOff', like_button.timesOff)
    else:
       thisExp.addData('like_button.timesOn', "")
       thisExp.addData('like_button.timesOff', "")
    thisExp.addData('dislike_button.numClicks', dislike_button.numClicks)
    if dislike_button.numClicks:
       thisExp.addData('dislike_button.timesOn', dislike_button.timesOn)
       thisExp.addData('dislike_button.timesOff', dislike_button.timesOff)
    else:
       thisExp.addData('dislike_button.timesOn', "")
       thisExp.addData('dislike_button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[cross_sign_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_sign_2* updates
        
        # if cross_sign_2 is starting this frame...
        if cross_sign_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_sign_2.frameNStart = frameN  # exact frame index
            cross_sign_2.tStart = t  # local t and not account for scr refresh
            cross_sign_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_sign_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_sign_2.started')
            # update status
            cross_sign_2.status = STARTED
            cross_sign_2.setAutoDraw(True)
        
        # if cross_sign_2 is active this frame...
        if cross_sign_2.status == STARTED:
            # update params
            pass
        
        # if cross_sign_2 is stopping this frame...
        if cross_sign_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6.0-frameTolerance:
                # keep track of stop time/frame for later
                cross_sign_2.tStop = t  # not accounting for scr refresh
                cross_sign_2.tStopRefresh = tThisFlipGlobal  # on global time
                cross_sign_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_sign_2.stopped')
                # update status
                cross_sign_2.status = FINISHED
                cross_sign_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_2.maxDurationReached:
        routineTimer.addTime(-break_2.maxDuration)
    elif break_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "great_wall_souvenir" ---
    # create an object to store info about Routine great_wall_souvenir
    great_wall_souvenir = data.Routine(
        name='great_wall_souvenir',
        components=[great_wall_souvenir_image],
    )
    great_wall_souvenir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for great_wall_souvenir
    great_wall_souvenir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    great_wall_souvenir.tStart = globalClock.getTime(format='float')
    great_wall_souvenir.status = STARTED
    thisExp.addData('great_wall_souvenir.started', great_wall_souvenir.tStart)
    great_wall_souvenir.maxDuration = None
    # keep track of which components have finished
    great_wall_souvenirComponents = great_wall_souvenir.components
    for thisComponent in great_wall_souvenir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "great_wall_souvenir" ---
    great_wall_souvenir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *great_wall_souvenir_image* updates
        
        # if great_wall_souvenir_image is starting this frame...
        if great_wall_souvenir_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            great_wall_souvenir_image.frameNStart = frameN  # exact frame index
            great_wall_souvenir_image.tStart = t  # local t and not account for scr refresh
            great_wall_souvenir_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(great_wall_souvenir_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'great_wall_souvenir_image.started')
            # update status
            great_wall_souvenir_image.status = STARTED
            great_wall_souvenir_image.setAutoDraw(True)
        
        # if great_wall_souvenir_image is active this frame...
        if great_wall_souvenir_image.status == STARTED:
            # update params
            pass
        
        # if great_wall_souvenir_image is stopping this frame...
        if great_wall_souvenir_image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 4.0-frameTolerance:
                # keep track of stop time/frame for later
                great_wall_souvenir_image.tStop = t  # not accounting for scr refresh
                great_wall_souvenir_image.tStopRefresh = tThisFlipGlobal  # on global time
                great_wall_souvenir_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'great_wall_souvenir_image.stopped')
                # update status
                great_wall_souvenir_image.status = FINISHED
                great_wall_souvenir_image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            great_wall_souvenir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in great_wall_souvenir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "great_wall_souvenir" ---
    for thisComponent in great_wall_souvenir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for great_wall_souvenir
    great_wall_souvenir.tStop = globalClock.getTime(format='float')
    great_wall_souvenir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('great_wall_souvenir.stopped', great_wall_souvenir.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if great_wall_souvenir.maxDurationReached:
        routineTimer.addTime(-great_wall_souvenir.maxDuration)
    elif great_wall_souvenir.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "questionnaire_2" ---
    # create an object to store info about Routine questionnaire_2
    questionnaire_2 = data.Routine(
        name='questionnaire_2',
        components=[questionnaire_2_text, like_button_2, dislike_button_2],
    )
    questionnaire_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset like_button_2 to account for continued clicks & clear times on/off
    like_button_2.reset()
    # reset dislike_button_2 to account for continued clicks & clear times on/off
    dislike_button_2.reset()
    # store start times for questionnaire_2
    questionnaire_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    questionnaire_2.tStart = globalClock.getTime(format='float')
    questionnaire_2.status = STARTED
    thisExp.addData('questionnaire_2.started', questionnaire_2.tStart)
    questionnaire_2.maxDuration = None
    # keep track of which components have finished
    questionnaire_2Components = questionnaire_2.components
    for thisComponent in questionnaire_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "questionnaire_2" ---
    questionnaire_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *questionnaire_2_text* updates
        
        # if questionnaire_2_text is starting this frame...
        if questionnaire_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            questionnaire_2_text.frameNStart = frameN  # exact frame index
            questionnaire_2_text.tStart = t  # local t and not account for scr refresh
            questionnaire_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(questionnaire_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'questionnaire_2_text.started')
            # update status
            questionnaire_2_text.status = STARTED
            questionnaire_2_text.setAutoDraw(True)
        
        # if questionnaire_2_text is active this frame...
        if questionnaire_2_text.status == STARTED:
            # update params
            pass
        # *like_button_2* updates
        
        # if like_button_2 is starting this frame...
        if like_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            like_button_2.frameNStart = frameN  # exact frame index
            like_button_2.tStart = t  # local t and not account for scr refresh
            like_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(like_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'like_button_2.started')
            # update status
            like_button_2.status = STARTED
            win.callOnFlip(like_button_2.buttonClock.reset)
            like_button_2.setAutoDraw(True)
        
        # if like_button_2 is active this frame...
        if like_button_2.status == STARTED:
            # update params
            pass
            # check whether like_button_2 has been pressed
            if like_button_2.isClicked:
                if not like_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    like_button_2.timesOn.append(like_button_2.buttonClock.getTime())
                    like_button_2.timesOff.append(like_button_2.buttonClock.getTime())
                elif len(like_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    like_button_2.timesOff[-1] = like_button_2.buttonClock.getTime()
                if not like_button_2.wasClicked:
                    # end routine when like_button_2 is clicked
                    continueRoutine = False
                if not like_button_2.wasClicked:
                    # run callback code when like_button_2 is clicked
                    pass
        # take note of whether like_button_2 was clicked, so that next frame we know if clicks are new
        like_button_2.wasClicked = like_button_2.isClicked and like_button_2.status == STARTED
        # *dislike_button_2* updates
        
        # if dislike_button_2 is starting this frame...
        if dislike_button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            dislike_button_2.frameNStart = frameN  # exact frame index
            dislike_button_2.tStart = t  # local t and not account for scr refresh
            dislike_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dislike_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dislike_button_2.started')
            # update status
            dislike_button_2.status = STARTED
            win.callOnFlip(dislike_button_2.buttonClock.reset)
            dislike_button_2.setAutoDraw(True)
        
        # if dislike_button_2 is active this frame...
        if dislike_button_2.status == STARTED:
            # update params
            pass
            # check whether dislike_button_2 has been pressed
            if dislike_button_2.isClicked:
                if not dislike_button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    dislike_button_2.timesOn.append(dislike_button_2.buttonClock.getTime())
                    dislike_button_2.timesOff.append(dislike_button_2.buttonClock.getTime())
                elif len(dislike_button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    dislike_button_2.timesOff[-1] = dislike_button_2.buttonClock.getTime()
                if not dislike_button_2.wasClicked:
                    # end routine when dislike_button_2 is clicked
                    continueRoutine = False
                if not dislike_button_2.wasClicked:
                    # run callback code when dislike_button_2 is clicked
                    pass
        # take note of whether dislike_button_2 was clicked, so that next frame we know if clicks are new
        dislike_button_2.wasClicked = dislike_button_2.isClicked and dislike_button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            questionnaire_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in questionnaire_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "questionnaire_2" ---
    for thisComponent in questionnaire_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for questionnaire_2
    questionnaire_2.tStop = globalClock.getTime(format='float')
    questionnaire_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('questionnaire_2.stopped', questionnaire_2.tStop)
    thisExp.addData('like_button_2.numClicks', like_button_2.numClicks)
    if like_button_2.numClicks:
       thisExp.addData('like_button_2.timesOn', like_button_2.timesOn)
       thisExp.addData('like_button_2.timesOff', like_button_2.timesOff)
    else:
       thisExp.addData('like_button_2.timesOn', "")
       thisExp.addData('like_button_2.timesOff', "")
    thisExp.addData('dislike_button_2.numClicks', dislike_button_2.numClicks)
    if dislike_button_2.numClicks:
       thisExp.addData('dislike_button_2.timesOn', dislike_button_2.timesOn)
       thisExp.addData('dislike_button_2.timesOff', dislike_button_2.timesOff)
    else:
       thisExp.addData('dislike_button_2.timesOn', "")
       thisExp.addData('dislike_button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "questionnaire_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_text],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # if end_text is stopping this frame...
        if end_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_text.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                end_text.tStop = t  # not accounting for scr refresh
                end_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_text.stopped')
                # update status
                end_text.status = FINISHED
                end_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
