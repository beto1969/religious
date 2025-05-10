#%%
import pandas as pd

#%%
df = pd.read_csv(r'C:\Users\mvped\Desktop\NLPProject\Code\Religious_Text_Dataset_Skeleton.csv', encoding='utf-8')

with open(r'C:\Users\mvped\Desktop\NLPProject\Code\Texts\Bhagavad_Gita.txt', 'r') as file:
    bhagavad_gita = file.read()

bhagavad_gita = bhagavad_gita.replace('\n', ' ')

with open(r'C:\Users\mvped\Desktop\NLPProject\Code\Texts\Book_of_Mormon.txt', 'r') as file:
    book_of_mormon = file.read()

book_of_mormon = book_of_mormon.replace('\n', ' ')

with open(r'C:\Users\mvped\Desktop\NLPProject\Code\Texts\DighaNikaya210710.txt', 'r') as file:
    digha_nikaya = file.read()

digha_nikaya = digha_nikaya.replace('\n', ' ')

with open(r'C:\Users\mvped\Desktop\NLPProject\Code\Texts\Genesis_Torah.txt', 'r') as file:
    genesis_torah = file.read()

genesis_torah = genesis_torah.replace('\n', ' ')

with open(r'Texts\kojiki.txt', 'r') as file:
    kojiki = file.read()

kojiki = kojiki.replace('\n', ' ')

with open(r'Texts\Popol_Vuh.txt', 'r') as file:
    popol_vuh = file.read()

popol_vuh = popol_vuh.replace('\n', ' ')

with open(r'Texts\Quran_Plain_Text.txt', 'r') as file:
    quran = file.read()

quran = quran.replace('\n', ' ')

with open(r'Texts\Siri Guru Granth - English Translation (matching pages).txt', 'r') as file:
    guru_grandth = file.read()

guru_grandth = guru_grandth.replace('\n', ' ')

with open(r'Texts\Tao_Te_Ching.txt', 'r') as file:
    tao_te_ching = file.read()

tao_te_ching = tao_te_ching.replace('\n', ' ')

with open(r'Texts\Zend_Avesta.txt', 'r') as file:
    zend_avesta = file.read()

zend_avesta = zend_avesta.replace('\n', ' ')


with open(r'Texts\Amazulu_Religion.txt', 'r') as file:
    Amazulu_Religion = file.read()

Amazulu_Religion = Amazulu_Religion.replace('\n', ' ')

with open(r'Texts\Ife_and_Yoruba_Myths.txt', 'r') as file:
    Ife_and_Yoruba_Myths = file.read()

ífè_and_Yoruba_Myths = Ife_and_Yoruba_Myths.replace('\n', ' ')

with open(r'Texts\Inca_Myths_and_Apu_Ollantay.txt', 'r') as file:
    Inca_Myths_and_Apu_Ollantay = file.read()

Inca_Myths_and_Apu_Ollantay = Inca_Myths_and_Apu_Ollantay.replace('\n', ' ')

with open(r'Texts\Mabinogian.txt', 'r') as file:
    Mabinogian = file.read()

Mabinogian = Mabinogian.replace('\n', ' ')

with open(r'Texts\Carib_Myths.txt', 'r') as file:
    Carib_Myths = file.read()

Carib_Myths = Carib_Myths.replace('\n', ' ')

with open(r'Texts\Prose_Edda.txt', 'r') as file:
    Prose_Edda = file.read()

Prose_Edda = Prose_Edda.replace('\n', ' ')

with open(r'Texts\Arawak_Myths.txt', 'r') as file:
    Arawak_Myths = file.read()

Arawak_Myths = Arawak_Myths.replace('\n', ' ')

with open(r'Texts\Maori_Myths.txt', 'r') as file:
    Maori_Myths = file.read()

Maori_Myths = Maori_Myths.replace('\n', ' ')

with open(r'Texts\Cherokee_Myths.txt', 'r') as file:
    Cherokee_Myths = file.read()

Cherokee_Myths = Cherokee_Myths.replace('\n', ' ')

with open(r'Texts\The_Book_of_The_Dead.txt', 'r') as file:
    The_Book_of_The_Dead = file.read()

The_Book_of_The_Dead = The_Book_of_The_Dead.replace('\n', ' ')

with open(r'Texts\Warao_Myths.txt', 'r') as file:
    Warao_Myths = file.read()

Warao_Myths = Warao_Myths.replace('\n', ' ')

with open(r'Texts\Chukchee_Myths.txt', 'r') as file:
    Chukchee_Myths = file.read()

Chukchee_Myths = Chukchee_Myths.replace('\n', ' ')

with open(r'Texts\Yuwaalaray_Myths.txt', 'r') as file:
    Yuwaalaray_Myths = file.read()

Yuwaalaray_Myths = Yuwaalaray_Myths.replace('\n', ' ')

with open(r'Texts\Hawaiian_Myths.txt', 'r') as file:
    Hawaiian_Myths = file.read()

Hawaiian_Myths = Hawaiian_Myths.replace('\n', ' ')

with open(r'Texts\Xhosa_Religion.txt', 'r') as file:
    Xhosa_Religion = file.read()

Xhosa_Religion = Xhosa_Religion.replace('\n', ' ')

with open(r'Texts\Greenlandic_Inuit_Myths.txt', 'r') as file:
    Greenlandic_Inuit_Myths = file.read()

Greenlandic_Inuit_Myths = Greenlandic_Inuit_Myths.replace('\n', ' ')

with open(r'Texts\Homeric_Hymns.txt', 'r') as file:
    Homeric_Hymns = file.read()

Homeric_Hymns = Homeric_Hymns.replace('\n', ' ')

with open(r'Texts\Igorot_Myths.txt', 'r') as file:
    Igorot_Myths = file.read()

Igorot_Myths = Igorot_Myths.replace('\n', ' ')

with open(r'Texts\Kebra_Nagast.txt', 'r') as file:
    Kebra_Nagast = file.read()

Kebra_Nagast = Kebra_Nagast.replace('\n', ' ')

#%%
print(df.head())

# %%
df.at[0, 'Text'] = bhagavad_gita
df.at[1, 'Text'] = book_of_mormon
df.at[2, 'Text'] = digha_nikaya
df.at[3, 'Text'] = genesis_torah
df.at[4, 'Text'] = kojiki
df.at[5, 'Text'] = popol_vuh
df.at[6, 'Text'] = quran
df.at[7, 'Text'] = guru_grandth
df.at[8, 'Text'] = tao_te_ching
df.at[9, 'Text'] = zend_avesta
df.at[10, 'Text'] = Amazulu_Religion
df.at[11, 'Text'] = ífè_and_Yoruba_Myths
df.at[12, 'Text'] = Inca_Myths_and_Apu_Ollantay
df.at[13, 'Text'] = Mabinogian
df.at[14, 'Text'] = Carib_Myths
df.at[15, 'Text'] = Prose_Edda
df.at[16, 'Text'] = Arawak_Myths
df.at[17, 'Text'] = Maori_Myths
df.at[18, 'Text'] = Cherokee_Myths
df.at[19, 'Text'] = The_Book_of_The_Dead
df.at[20, 'Text'] = Warao_Myths
df.at[21, 'Text'] = Chukchee_Myths
df.at[22, 'Text'] = Yuwaalaray_Myths
df.at[23, 'Text'] = Hawaiian_Myths
df.at[24, 'Text'] = Xhosa_Religion
df.at[25, 'Text'] = Greenlandic_Inuit_Myths
df.at[26, 'Text'] = Homeric_Hymns
df.at[27, 'Text'] = Igorot_Myths
df.at[28, 'Text'] = Kebra_Nagast
# %%

print(df.head(40))
# %%
df.to_csv('Dataset_with_Text.csv', encoding='utf-8')
# %%
