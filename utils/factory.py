def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "bic":
        from models.bic import BiC
        return BiC(args)
    elif name == "lwf":
        from models.lwf import LwF
        return LwF(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "finetune":
        from models.finetune import Finetune
        return Finetune(args)
    elif name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "foster":
        from models.foster import FOSTER
        return FOSTER(args)
    
    #w infer models
    elif name == "concept1":
        from models.concept1 import concept1
        return concept1(args)
    elif name == "icarlwinfer":
        from models.icarlwinfer import iCaRL_winfer
        return iCaRL_winfer(args)
    elif name == "finetunewinfer":
        from models.finetunewinfer import Finetune_winfer
        return Finetune_winfer(args)
    elif name == "replaywinfer":
        from models.replaywinfer import Replay_winfer
        return Replay_winfer(args)
    elif name == "derwinfer":
        from models.derwinfer import DER_winfer
        return DER_winfer(args)
    elif name == 'fosterwinfer':
        from models.fosterwinfer import FOSTER_winfer
        return FOSTER_winfer(args)
    else:
        assert 0
