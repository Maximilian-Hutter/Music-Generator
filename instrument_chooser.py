from instruments import set_instrument_true
import torch 
import torch.nn as nn


class InstrumentChooserNet(nn.Module):    # gztan dataset or FMA dataset or openMIC prepare data with Predominant Musical Instrument Classification
    def __init__(self):
        super(InstrumentChooserNet, self).__init__()   

        self.number_of_instruments = number_of_instruments

    def forward(self, input):
        instrument_indexes = []

        
        for i in self.number_of_instruments:
            instrument_indexes.append(out_instrument)

        return instrument_indexes




# synth brass, synth organ, synth string, ac synthlead, ec synthlead always False!