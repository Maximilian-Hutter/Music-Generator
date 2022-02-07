

class Instruments:
    def __init_(self, source, family, index, state):
        self.source = source
        self.family = family
        self.index = index
        self.state = state

instruments_list = []



instruments_list.append( Instruments("AC", "Bass", 0, False))
instruments_list.append( Instruments("EC","Bass",1,False))
instruments_list.append( Instruments("SYNTH", "Bass",2,False))
instruments_list.append( Instruments("AC", "Brass",3,False))
instruments_list.append( Instruments("EC", "Brass",4,False))
instruments_list.append( Instruments("SYNTH", "Brass",5,False))
instruments_list.append( Instruments("AC", "Flute",6,False))
instruments_list.append( Instruments("EC", "Flute",7,False))
instruments_list.append( Instruments("SYNTH", "Flute",8,False))
instruments_list.append( Instruments("AC", "Guitar",9,False))
instruments_list.append( Instruments("EC", "Guitar",10,False))
instruments_list.append( Instruments("SYNTH", "Guitar",11,False))
instruments_list.append( Instruments("AC", "Keyboard",12,False))
instruments_list.append( Instruments("EC", "Keyboard",13,False))
instruments_list.append( Instruments("SYNTH", "Keyboard",14,False))
instruments_list.append( Instruments("AC", "Mallet",15,False))
instruments_list.append( Instruments("EC", "Mallet",16,False))
instruments_list.append( Instruments("SYNTH", "Mallet",17,False))
instruments_list.append( Instruments("AC", "Organ",18,False))
instruments_list.append( Instruments("EC", "Organ",19,False))
instruments_list.append( Instruments("SYNTH", "Organ",20,False))
instruments_list.append( Instruments("AC", "Reed",21,False))
instruments_list.append( Instruments("EC", "Reed",22,False))
instruments_list.append( Instruments("SYNTH", "Reed",23,False))
instruments_list.append( Instruments("AC", "String",24,False))
instruments_list.append( Instruments("EC", "String",25,False))
instruments_list.append( Instruments("SYNTH", "String",26,False))
instruments_list.append( Instruments("AC", "Synthlead",27,False))
instruments_list.append( Instruments("EC", "Synthlead",28,False))
instruments_list.append( Instruments("SYNTH", "Synthlead",29,False))
instruments_list.append( Instruments("AC", "Vocal",30,False))
instruments_list.append( Instruments("EC", "Vocal",31,False))
instruments_list.append( Instruments("SYNTH", "Vocal",32,False))
                

def set_instruments(instrument_index):
    using_instruments = []

    for i in instruments_list:
        for a in instrument_index:
            if instruments_list[i].index == instrument_index[a]:
                instruments_list[i].state = True
                using_instruments.append = instruments_list[i]

    return using_instruments