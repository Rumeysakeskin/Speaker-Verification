from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java

ZEMBEREK_PATH ='C:/Users/Dell/Desktop/beyzaDosyalar/Python/Sentiment Analiz/zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
#Lemmatizasyon
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()
TurkishSpellChecker: JClass = JClass('zemberek.normalization.TurkishSpellChecker')
spell_checker: TurkishSpellChecker = TurkishSpellChecker(morphology)


def spellChecker(tokens):
    for index, token in enumerate(tokens):

        # yazım yanlısı varsa if'e girer
        if not spell_checker.check(JString(token)):

            if spell_checker.suggestForWord(JString(token)):
                # kelimenin doğru halini döndürür.
                tokens[index] = spell_checker.suggestForWord(JString(token))[0]
                # print((spell_checker.suggestForWord(JString(token))[0]))

    # Java liste yapısı listeye eklenerek düzeltilir.
    corrected = [str(i) for i in tokens]

    return " ".join(corrected)


exmp = "yazımığ okuduğunuüz içn teşekkr edrim"
print(spellChecker(exmp.split()))
