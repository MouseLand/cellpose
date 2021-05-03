from .io import logger_setup
try:
    from cellpose import gui 
    GUI_ENABLED = True 
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    
logger, log_file = logger_setup()
