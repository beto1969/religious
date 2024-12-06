#%%
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from collections import defaultdict

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

#%%

stop_words = set(stopwords.words('english'))
additional_stop_words = ['arjuna', 'krishna', 'mazda', 'ahura mazda', 'brahma', 'dhammas', 'moroni',
                'jacob', 'enos', 'christ', 'unas', 'osiris', 'mosiah', 'prabhaatee mehl', 'ani',
                'text chapter', 'jared', 'brother jared', 'pharoah', 'moses', 'mallans', 'kusinr', 
                'coriantumr', 'shiz', 'pas', 'came pas', 'lib', 'qasiagssaq', 'ven', 'nanda',
                'ven nanda', 'tathgata', 'dadda', 'nanna', 'raag fifth', 'siree raag', 'dayv',
                'naam dayv', 'naam', 'arit', 'dhamma', 'shabad', 'maajh', 'vajjians', 'nanda', 
                'magadha', 'vassakra', 'jvaka', 'saarang fourth', 'shalok', 'pauree', 'pukkusa', 
                'pv', 'avesta', 'zend', 'sraosha', 'chukchee', 'eskimo', 'iii', 'fargard', 'fargards',
                'venddd', 'farg', 'abraham', 'lut', 'isaac', 'noah', 'armf', 'orsha', 'hermes', 'apollo', 
                'odin', 'njrdr', 'skadi', 'xibalba', 'hunahp', 'xbalanqu', 'yawahu', 'tohil',
                'tohil avilix', 'mahucutah', 'hacavitz', 'avilix', 'aviliz hacavitz', 'dava',
                'thrita', 'zipacn', 'cabracn', 'hunahp xbalanqu', 'dinewan', 'goombelgubbon', 'oolah',
                'dinewans', 'ha', 'wayambeh', 'galah', 'kgssagssuk', 'little kgssagssuk', 'little umerdlugtoq',
                'ra', 'salih', 'arthur', 'gangleri', 'thor', 'skrmir', 'hrungnir', 'hymir',
                'allah', 'son mary', 'mary', 'jesus', 'jesus son', 'allah lord', 'zeus', 
                'maia', 'son maia', 'son zeus', 'mus', 'dy', 'kabeer', 'inkalimeva', 'fareed',
                'mehl fareed', 'lord fareed', 'gir', 'metaneira', 'demeter', 'mugals', 'gauree',
                'raag mehl', 'sahu', 'pepi', 'ka', 'bilaaval', 'dhanaasaree fifth', 'dhanaasaree',
                'second mehl', 'lord dakhanay', 'pauree', 'hukam', 'todee fifth', 'todee',
                'vii', 'viii', 'asura', 'varuna', 'tuyallay', 'ptah', 'horus', 'maat', 'set', 'shu',
                'nut', 'seb', 'zarathustra', 'unkulunkulu', 'amadhlozi', 'amatongo', 'umancele',
                'usopetu', 'upeteni', 'itongo', 'ufaku', 'uncapayi', 'udumisa', 'rangitu',
                'kauilani', 'maui', 'pikoi', 'mainele', 'tabu', 'umyeka', 'arawak', 'ukaleq', 
                'atdlarneq', 'lumawig', 'konehu', 'll', 'll ll', 'avvang', 'neruvkq', 'sigurdr',
                'mamala', 'ouha', 'dkw', 'deereeree', 'wyah', 'bibbee', 'kamapuaa', 'pele',
                'ollantay', 'tupac yupanqui', 'tupac', 'yupanqui', 'uillac uma', 'osiris ani',
                'konehu', 'qujvrssuk', 'tupilak', 'tugto', 'pia', 'carib', 'makusis', 'arawak',
                'tutanekai', 'te', 'kawelo', 'kauai', 'namaka', 'aiyobanni', 'hatupatu', 'siwara', 
                'mawri', 'okoyumo', 'papik', 'serikoai', 'konehu', 'koneso', 'deegeenboyah', 'mullyan',
                'mullyangah', 'ah ah', 'te ponga', 'ponga', 'tawhaki', 'karihi', 'tatua', 'weedah',
                'weeoombeens', 'piggiebillah', 'bougoodoogahdah', 'bahloo', 'turi', 'arawa', 
                'manoa', 'oahu', 'nuuanu', 'kou', 'whakatau', 'seneca', 'lawrence', 'asalq', 'makte',
                'sedna', 'guillemot', 'takarangi', 'hakawau', 'puarata', 'maketu', 'komatari',
                'kororomanna', 'hebu', 'koneso', 'nafudi', 'haburi', 'hlakanyana', 'sikulume',
                'inabulele', 'mangangezulu', 'ptussorssuaq', 'altaq', 'qalagnguas', 'tallarssuaq',
                'kupe', 'turi', 'hoturapa', 'rehua', 'rupe', 'manaia', 'ihenga', 'wakea', 'hervey',
                'kamapuaa', 'kokoa', 'awa', 'manoa', 'kapuni', 'dhanna', 'gobind', 'lludd', 'isis',
                'ahura', 'nephi', 'nephites', 'giddianhi', 'gidgiddoni', 'lachoneus', 'gunnarr', 'brynhildr', 
                'gunnar hgni', 'hgni', 'gudrn', 'atli', 'gjki', 'limhi', 'ammon', 'zarahelma', 'king limhi', 
                'jvaka', 'inuence', 'pali', 'khadoor', 'pheru', 'elphin', 'taliesin', 'heinin', 'maelgwn', 
                'owain', 'kynon', 'kai', 'owain kynon', 'alma', 'benet', 'gurmukh', 'nanda', 'sagha', 'ambapl',
                'licchavis', 'bairaaree', 'raam', 'har har', 'kusinr', 'tathgata', 'sal', 'armf', 'orsha', 
                'odudwa', 'odwa', 'haungaroa', 'ueneku', 'potikiroroa', 'ku', 'heb', 'laban', 'hunahp', 'agusinnguaq',
                'knagssuaq', 'zipacn', 'cabracn', 'hunahp xbalanqu', 'xbalanqu', 'piqui', 'chaqui', 'piqui chaqui', 
                'coyllur', 'loki', 'idunn', 'troolie', 'luqman', 'tao', 'venddd', 'prabhaatee', 'basant', 'basant mehl', 
                'zoroaster', 'pahlavi', 'nanak', 'yasna', 'asha', 'baresman', 'master asha', 'malaar fifth', 'malaar',
                'maya', 'aasaa', 'intelezi', 'vara', 'goojaree', 'yima', 'uthlanga', 'utikxo', 'isalukazana', 'hrlfr', 
                'thjazi', 'ymir', 'kumagdlak', 'james', 'ornyan', 'bo', 'adaba', 'insingizi', 'hebus', 'mrimi', 'bo', 
                'moremi', 'ayar', 'viracocha', 'ccapac', 'manco', 'manco ccapac', 'shawano', 'gwyn', 'anarteq', 'kardltuarssuk', 
                'helen', 'vatea', 'llevelys', 'lludd', 'coranians', 'caridwen', 'einarr', 'einarr sang', 'gylfi', 
                'brokkr', 'owain', 'caribs', 'hreidmarr', 'freyr', 'whakaue', 'hotunui', 'wurrunnunnah', 'wirreenun', 
                'bunnyyarl', 'noondoo', 'byamee', 'ptussorssuaq', 'hariwali', 'asalq', 'arawaks', 'caribs', 'pomeroon', 
                'bootoolgah', 'goonur', 'comebee', 'bootoolgah goonur', 'corrobboree', 'ite', 'kurreahs', 'narran', 
                'byamee', 'haumea', 'tamure', 'kiki', 'olopana', 'goomblegubbon', 'goomblegubbons', 'irawaru', 'paka', 
                'kahureremoa', 'aotea', 'fakalan', 'kanawa', 'kapas', 'kgssagssuk', 'little kgssagssuk', 'umerdlugtoq', 
                'goonur', 'goonur husband', 'isigligrssik', 'chi', 'chi chi', 'birrahlee', 'wahroogahs', 'bunbundoolooey',
                'bangan', 'qujvrssuk', 'kapalama', 'amerdloq', 'simo', 'ebbong', 'corial', 'piai', 'gwineeboo', 'ah ah',
                'gidgereegah', 'aaron', 'amalekites', 'amulek', 'zarathushtra', 'korihor', 'adam', 'pharoah', 'iblis',
                'inkosazana', 'mormon', 'helaman', 'kishkumen', 'morianton', 'pahoran', 'amulek', 'zeezrom', 'judah', 
                'nips', 'cf', 'tamar', 'gond', 'mrimi', 'bo', 'shu', 'lamoni', 'cachi', 'kaw', 'narahdarn', 'surtr',
                'joseph', 'nanda', 'jee', 'yaya', 'brahmin', 'kaanraa', 'tilang', 'gurmukhs', 'benet', 'raamkalee',
                'mehl', 'baynee', 'lohicca', 'rosika', 'slavatik', 'rosika barber', 'saadh', 'saadh sangat',
                'fifth mehl', 'saarang fifth', 'saarang', 'fifth mehl', 'ephron', 'sarah', 'abimelech', 'freyja',
                'abram', 'sodom', 'zoar', 'iwa', 'umi', 'isigligrssik', 'galagnguas', 'rachel', 'leah', 'gooloo',
                'comebees', 'esau', 'rebekah', 'usigwili', 'amasi', 'umkqaekana', 'amazulu', 'armf', 'orsha', 'odudwa',
                'odwa', 'zipacn', 'cabracn', 'hunahp xbalanqu', 'xbalanqu', 'hunahp', 'mehl', 'inyanga', 'hai', 'impepo',
                'hai hai', 'oom', 'oom oom', 'wurrunnah', 'dayoorls', 'ukoko', 'ukulukulu', 'udhlamini', 'unsondo',
                'banab', 'maraka', 'kakuhihewa', 'gir', 'bragi', 'ubulawo', 'indras', 'shivas', 'mehl', 'second mehl',
                'uillac uma', 'uma', 'uillac', 'ravi', 'ravi daas', 'daas', 'bairaagi', 'soohee jee', 'nasu', 
                'balwand', 'khivi', 'ffnir', 'reginn', 'hgni', 'gudrn', 'jrmunrekkr', 'erpr', 'king jrmunrekkr', 'srli',
                'hamdir', 'kohora', 'moogaray', 'eehu', 'masilo', 'kenkebe', 'kardltuarssuk', 'raumati', 'karika',
                'kakei', 'tinirau', 'kae', 'tatau', 'amanxusa', 'rhiannon', 'pryderi rhiannon', 'pryderi', 'cantrevs',
                'ornyan', 'armf', 'nyanribo', 'yawarri', 'qujvrssuk', 'isokun', 'iddawc', 'beeargah', 'borah',
                'bendigeid', 'bendigeid vran', 'vran', 'branwen', 'matholwch', 'branwen', 'kae', 'tinirau', 'tutunui',
                'hrr', 'leto', 'genii', 'rua', 'tama', 'rata', 'kanaloa', 'aikanaka', 'kakuhihewa', 'kapoi', 'wabassi',
                'nuknguasik', 'kgssagssuk', 'little kgssagssuk', 'dkw', 'gatan', 'boliwan', 'ideo', 'anitos',
                'qujvrssuk', 'warribisi', 'moodai', 'paiwarri', 'kokerite', 'maikoha', 'ngatora', 'kahukura',
                'whatuiapiti', 'heimdallr', 'dionysus', 'makunaima', 'ornyan', 'bo', 'olbo', 'thjlfi', 'peredur',
                'mrimi', 'oluronbi', 'bamu', 'hioi', 'ptussorssuaq', 'yurokon', 'nat', 'nat fifth', 'dkw', 'kgssagssuk',
                'isigligrssik', 'hou', 'hawepotiki', 'uenuku', 'prahlaad', 'harnaakhash', 'ouyan', 'yuckay', 'yuckay yuckay',
                'comebo', 'cronos', 'ah', 'ah ah', 'ooboon', 'wh', 'whn', 'solomon', 'hunahp', 'mahthi', 'norns', 'hrr',
                'nk', 'njps', 'shechem', 'hamor', 'magahar', 'benares', 'bhairao', 'gwaarayree', 'waaho', 'hindol',
                'lord hindol', 've', 'nanda', 'sagha', 'tathgata', 'midgard', 'maxen', 'lamanites', 'zion', 'armf',
                'orsha', 'odwa', 'odudwa', 'zoramites', 'jershon', 'gadianton', 'jarom', 'lamanites', 'ammoron', 'antipus',
                'deepak', 'maalakausak', 'chaytee', 'lalo', 'gwydion', 'gronw', 'zion', 'umwathleni', 'zarahelma',
                'peredur', 'luned', 'yma sumac', 'yma', 'hrr', 'danom', 'gabi', 'gomotan', 'ponaturi', 
                'kgssagssuk', 'little kgssagssuk', 'takakopiri', 'dkw', 'oriyu', 'asalq', 'uktena', 
                'uktena', 'maipuri', 'wawaiya', 'tuwhakararo', 'warraus', 'kaupe', 'pwyll', 'teirnyon',
                'heveydd', 'gwawl', 'gwalchmai', 'geraint', 'dummerh', 'mooregoo', 'gwai', 'bilbers',
                'hauraki', 'whatu', 'ngatoro', 'soulbride', 'kavi', 'harimandir', 'noid', 'astivihad',
                'gazi', 'goolay', 'rhonabwy', 'armf', 'orsha', 'odudwa', 'odwa', 'gir', 'sita lachhman',
                'hrr', 'har', 'mabon', 'llew', 'son modron', 'modron', 'mabon son', 'hrr', 'anurruddha',
                'devats', 'aairs', 'aogemadaeca', 'kusinr', 'tathgata', 'baldr', 'svadilfari', 'sagha',
                'sagha monks', 'ambapl', 'ongkaar', 'tathgata', 'bairaagan', 'gideon', 'har', 
                'moronihah', 'zarahelma', 'sidon', 'king laman', 'amlicites',  'iaen', 'eri', 'greid', 'greid son',
                'shule', 'son eri', 'corihor', 'kib', 'akish', 'mazdayasnians', 'thravan', 'athravan', 'myazda', 'zaotar',
                'sherrizah', 'tet', 'middoni', 'benjamin', 'king benjamin', 'amulon', 'nemmes', 'xbalanqu', 'hunahp xbalanqu',
                'hunahp', 'zerahemnah', 'ornyan', 'bo', 'olbo', 'simbukumbukwana', 'mbulu', 'skrnir', 'frigg',
                'asalq', 'ffnir', 'qujvrssuk', 'geirrdr', 'grdr', 'erim', 'erim', 'twrch', 'gweir', 'boku',
                 'boku boku', 'severn', 'grugyn', 'jtunheim', 'annwvyn', 'prince dyved', 'ynwyl', 'etlym',
                 'kgssagssuk', 'little kgssagssuk', 'nahakoboni' 'qalagnguas', 'kardltuarssuk', 'ornyan',
                 'bo', 'olbo', 'armf', 'orsha', 'odwa', 'odudwa', 'pasnush', 'hana', 'zaurura', 'usithlanu',
                 'wadahans', 'manmukhs', 'yas', 'gthas', 'david', 'en', 'zipacn', 'toi', 'manawyddan',
                 'sty', 'powys', 'mathonwy', 'hrr', 'aairs', 'kusinr', 'tathgata', 'raag bihaagraa',
                 'bihaagraa', 'manmukh', 'aasaavaree', 'subhadda', 'nanda', 'ahurian', 'ganges', 'soohee',
                 'soohee fifth', 'sagha', 'ndik', 'nanda', 'igbos', 'mrimi', 'oranyan', 'isaiah', 'laman',
                 'knigseq', 'qalagnguas', 'isigligrssik', 'asalq', 'hatcinodo', 'ga na', 'ga', 'kgssagssuk',
                 'little kgssagssuk', 'qujvrssuk', 'pakuanui', 'huhuti', 'weeoombeen', 'wauke', 'popohorokewa',
                 'tamanoa', 'kaakau', 'llwyddeu', 'gwadyn', 'pehu', 'leho', 'gudrn', 'frdi', 'hgni', 'gjki',
                 'makanauro', 'trwyth', 'llwydawg', 'kalma', 'chhant', 'orm', 'azi', 'xix', 'mainyu',
                 'angra mainyu', 'angra', 'introd', 'yast', 'maalaa', 'maalaa fifth', 'kapo', 'kalihi',
                 'rhun', 'kaydaaraa', 'gwenhwyvar', 'edeyrn', 'enid', 'cusi', 'umdabuko', 'rened', 'nanda',
                 'tathgata', 'sagha', 'gwyddno', 'ishmael', 'gwenhwyvar', 'glewlwyd', 'ffnir', 'hermdr', 'hel',
                 'abinadi', 'naglfar', 'hrr', 'sidom', 'musan', 'meamei', 'qujvrssuk', 'hoahanau', 'ipukai',
                 'halemanu', 'saunikoq', 'zim', 'suttungr', 'ffnir', 'baugi', 'nongwes', 'magoda', 'dardurr',
                 'willgoo', 'evnissyen',  'lono', 'ano', 'scr', 'ii', 'tangaroa', 'umahaule', 'unqanqaza',
                 'matahorua', 'kuramarotini', 'amanthlwenga', 'umdhlebe', 'neruvkq', 'avvang', 'dyved',
                 'imamba', 'guluwe', 'hili', 'nand', 'raavan', 'wh', 'dayoorl', 'wh wh', 'whn', 'tangalimlibo',
                 'lehna', 'druj', 'cain', 'abel', 'armf', 'orsha', 'isigligrssik', 'qalagnguas', 'bilaawal',
                 'ood', 'mukanday', 'jrmunrekkr', 'frdi', 'omer', 'jvaka', 'inuence', 'sebus', 'melchizedek',
                 'aphrodite', 'anchises', 'konane', 'marabuntas', 'heiau', 'baddasan', 'celeus', 'cowee',
                 'hinai', 'pisi', 'hatcinodo', 'sgard', 'skrmir', 'nanyobo', 'cantrev', 'blodeuwedd',
                 'frdi', 'gwythyr', 'nudd', 'gwythyr son', 'angusinnguaq', 'dkw', 'vishtaspa', 'king vishtaspa',
                 'nabnazdistas', 'amrit', 'sita lachhman', 'lahore', 'lachhman', 'raag gujri', 'raag',
                 'yasht', 'drvaspa', 'kavis', 'undhlebekazizwa', 'umazwana', 'amakuza', 'ifr', 'uggason','ifr uggason',
                 'cabracn', 'madawc', 'tegid', 'bach', 'gwion bach', 'iorwerth', 'hunbatz', 'hunchoun', 'hunbatz hunchoun',
                 'mrimi', 'bo', 'hrlfr', 'edom', 'gujri fifth', 'gujri', 'kevaa', 'trilochan', 'veda', 'dakhmas',
                 'vi', 'tishtrya', 'goolahgool', 'angusinnguaq', 'kapa', 'wh', 'whn', 'bindeah', 'uhu',
                 'kauahoa', 'keaau', 'hauula', 'tarbaran', 'grdr', 'frdi', 'geirrdr', 'dungle', 'hrr',
                 'skrmir', 'knigseq', 'forth spitama', 'spitama', 'jahi', 'sannyaasi', 'blverkr', 'tr',
                 'hrr', 'mrimi', 'bo', 'ifes', 'apakura', 'rongotakawiu', 'dinah', 'quich', 'dhadha',
                 'devats', 'kusinr', 'jaijaavantee', 'jaijaavantee ninth', 'rened', 'jvaka', 'ajtasattu', 
                 'samiri', 'nahakoboni', 'waiamari', 'skdbladnir', 'ayo', 'maalee', 'gauraa', 'maalee gauraa',
                 'gauraa fifth', 'mithra', 'bhagaautee', 'vaishnaav', 'kalyaan', 'poorbee fourth',
                 'tamub', 'menw', 'ermid', 'shu', 'gunas', 'eing', 'gotama', 'sushmanaa', 'ambapl',
                 'nanda', 'cundal', 'vinaya', 'dakhanay fifth', 'lord dakhanay', 'dakhanay', 'maru', 'cumorah',
                 'helam', 'jvaka', 'ajtasattu', 'vassakra', 'amlici', 'aurvandill', 'arnrr', 'eilfr', 'zarahelma',
                 'erbin', 'rangitihi', 'tupenu', 'nuknguasik', 'hatcinodo', 'ch', 'hrlfr', 'kraki',
                 'hrlfr kraki', 'adils', 'gudrn', 'tane', 'zakariya', 'hgni', 'hedinn', 'nkws', 'phoebus',
                 'idzumo', 'brahm', 'benet', 'udaasee', 'nabnazdistas', 'seq seq', 'istrs', 'innite',
                 'yspaddaden penkawr', 'penkawr', 'yspaddaden', 'gwrnach', 'potoru', 'tuau', 'goug', 'gour gah', 'gour gour',
                 'gah gah', 'tahiti', 'ngahue', 'hawaiki', 'tainui', 'ihuatamai', 'hinauri' 'ihuwareware', 'yackman', 'whakaturia',
                 'amanna', 'artuk', 'ptussorssuaq', 'altaq', 'gwion', 'mawri', 'birnie', 'ullr', 'sark', 'rdi', 'hildr', 'hlkk',
                 'hildr', 'hgni', 'kvasir', 'skald', 'ifr', 'eyvindr', 'thrvaldi', 'hrr', 'whn', 'kalaniopuu', 'mayrah',
                 'idhlozi', 'wa', 'asalq', 'ngngjuk', 'nuknguasik', 'qalagnguas', 'isigligrssik', 'felin', 'punihuia', 'kushi',
                 'armorica', 'gillingr', 'blverkr', 'geirrdr', 'fedilizan', 'kilauea', 'seq', 'istrs', 'nabnazdistas',
                 'anquetil', 'saawan', 'irth', 'krishnas', 'vishtasp', 'visperad', 'vohu', 'rashnu', 'verethraghna'
                 'ersonified', 'ndying', 'ndying eyond', 'atred', 'eyond', 'reative', 'samaadhi', 'govind', 'haray', 'lord haray',
                 'haray haray', 'aweoweo', 'frdi', 'kuiwai', 'drengs', 'haraldr', 'cotuh', 'ahpop', 'nihaib', 'eepa', 'kearoa',
                 'mahina', 'waolani', 'kahiki', 'llyr', 'caradawc', 'harlech', 'kynan', 'reuel', 'chedorlaomer',
                 'dishon', 'basemath', 'oholibamah', 'zibeon', 'anah', 'eliphaz', 'eos', 'gandhaaree', 'raaginis', 'hagar',
                 'hera', 'orsha', 'brahm', 'great brahm', 'maghar', 'katak', 'phalgun', 'saawan', 'bhaadon', 'assu', 'maagh',
                 'ambapl', 'pv', 'cunda', 'vesl', 'arezra', 'brahmans', 'nite', 'innite', 'nite innite', 'augustness',
                 'salla', 'pitu', 'pitu salla', 'kusinr', 'nanda', 'lamech', 'nahor', 'teancum', 'haran', 'terah', 'seth',
                 'rangi', 'rangi papa', 'ahr', 'zarahelma', 'ynywl', 'azhi', 'dava', 'daeva', 'thravan', 'saddar hyde', 'saddar',
                 'ephraim', 'bethuel', 'lehi', 'baresma', 'utshaka', 'goolays', 'usenzangakona', 'uyegana', 'uxele', 'unsikana',
                 'ulangeni', 'umpengula', 'mbanda', 'umpengula mbanda', 'uthlomo', 'ammaron', 'senum', 'kamoiliili', 'arnrr',
                 'eilfr', 'boki', 'fridleifr', 'amalickiah', 'teancum', 'lehi', 'pershephone', 'sanjaya', 'drona', 'bhishma',
                 'skrmir', 'ukanzi', 'sgard', 'goore', 'goore goore', 'mirrieh', 'saxland', 'eudav', 'gathas', 'gir',
                 'polack', 'gomez', 'vanir', 'vaar', 'gursikhs', 'gorakh', 'puraanas', 'qazi', 'blerwm', 'yamato',
                 'naad', 'bragr', 'hrr', 'valhall', 'urdr', 'vlusp', 'dathyl', 'caer dathyl', 'custennin', 'hilu',
                 'bela', 'saul', 'cyllenian', 'makte', 'bedwyr', 'sarai', 'vag', 'airyana', 'airyana vag', 'frangrasyan',
                 'aryan', 'aredvi', 'airyaman', 'saoka', 'pali', 'ajtasattu', 'ajtasattu vedhiputta', 'vedehiputta',
                 'sunidha', 'sunidha vassakra', 'vassakra', 'aad', 'aad mam', 'mam tanvo', 'mam', 'ithyejanguhaiti',
                 'gtha', 'tanvo ithyejanguhaiti', 'tanvo', 'kusinr', 'mah kassapa', 'lakshmi', 'king ajtasattu',
                 'ajtasattu', 'nigaha', 'pendaran', 'dava', 'ashi', 'ashi vanguhi', 'vanguhi', 'dakhma', 'bheekhan', 'dwaarikaa',
                 'teomner', 'manti', 'lehonti', 'emer', 'thjlfi', 'grjtnagard', 'zemnarihah', 'mulek', 'geirrdr', 'arianrod',
                 'mspell', 'hestia', 'gatha', 'kinvad', 'gtha', 'karshvares', 'strabo', 'venddd', 'verethraghna', 'verethraghna verethraghna',
                 'mohinis', 'kahilona', 'gilvaethwy', 'gwynedd', 'gilvaethwy son', 'sijjin', 'kilitraq', 'hiiakas', 'leurs', 'gmz',
                 'comm', 'ne', 'draona', 'gahi', 'gucumatz', 'synjur', 'beli', 'hermdr', 'hunchoun', 'isiwandiye', 'hina',
                 'ndik', 'ew', 'aairs', 'pairika', 'keresspa', 'vedic', 'dahka', 'vedas', 'aminadab', 'mazdeism', 'king ajtasattu',
                 'kassapa', 'shiblom', 'yggdrasill', 'hrr', 'vlusp', 'bifrst', 'wh', 'whn', 'wh wh', 'wondah', 'amos',
                 'senine', 'ormazd', 'pitris', 'haurvatt amerett', 'baisakhi', 'hud', 'qibla', 'kaashi', 'grani', 'niflungs',
                 'hgni', 'fri', 'hjadnings', 'hrlfr', 'gjki', 'camlan', 'echel', 'son saidi', 'saidi', 'gwallt', 'refr', 'sif',
                 'blverkr', 'kamehameha', 'waikiki', 'waipio', 'ammonihah', 'melek', 'shiblon', 'sigyn', 'frbauti', 'jtunheim',
                 'thravan', 'havgan', 'arawn', 'kumaso', 'dhritirashtra', 'saadhus', 'vedas']
combined_stop_words = stop_words.union(additional_stop_words)
lemmatizer = WordNetLemmatizer()


class ReligiousTextAnalyzer:
    def __init__(self, csv_path):
        # initialize the analyzer with the CSV file
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.documents = self.df['Text'].tolist()
        self.names = self.df['Name_of_Text'].tolist()
        self.topics_by_document = {}
        self.topic_similarities = None

    def preprocess(self, text):
        text = re.sub(r'\W', ' ', text)
        words = nltk.word_tokenize(text.lower())
        filtered_words = [lemmatizer.lemmatize(word) for word in words
                          if word.isalpha() and word not in combined_stop_words]
        return filtered_words

    def generate_ngrams(self, texts, n=3):

        bigram = Phrases(texts, min_count=3, threshold=10)
        trigram = Phrases(bigram[texts], threshold=10)
        texts = [trigram[bigram[text]] for text in texts]
        filtered_texts = [[word for word in text if word not in combined_stop_words
                           and '_' not in word] for text in texts]
        return filtered_texts

    def extract_topics(self):
        """extract topics using LDA approach"""
        for i, document in enumerate(self.documents):
            processed_doc = self.preprocess(document)
            processed_doc = self.generate_ngrams([processed_doc])[0]

            dictionary = corpora.Dictionary([processed_doc])
            corpus = [dictionary.doc2bow(processed_doc)]

            lda_model = models.LdaModel(corpus, num_topics=5,
                                        id2word=dictionary, passes=15)

            topics = []

            # extract top 10 words for each topic
            for idx, topic in lda_model.print_topics(-1): # -1 means all topics
                topic_words = [word.split('*')[1].strip('"')
                               for word in topic.split(' + ')]
                topic_words = [word for word in dict.fromkeys(topic_words)
                               if word not in combined_stop_words][:10]
                while len(topic_words) < 10:
                    topic_words.append("")
                topics.append(', '.join(topic_words))

            self.topics_by_document[self.names[i]] = topics
        return self.topics_by_document

    def calculate_topic_similarities(self):

        """calculate similarity matrix between documents based on their topics"""
        texts = list(self.topics_by_document.keys())
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))

        # create a flat list of all topics for each text
        text_topics = {text: ' '.join(topics)
                       for text, topics in self.topics_by_document.items()}

        # calculate tf-idf vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_topics.values())

        # calculate cosine similarities
        for i in range(n_texts):
            for j in range(n_texts):
                if i != j:
                    similarity = 1 - cosine(
                        tfidf_matrix[i].toarray().flatten(),
                        tfidf_matrix[j].toarray().flatten()
                    )
                    similarity_matrix[i, j] = similarity

        self.topic_similarities = pd.DataFrame(
            similarity_matrix,
            index=texts,
            columns=texts
        )
        # return the similarity matrix
        return self.topic_similarities

    def analyze_environmental_terms(self, environmental_categories):
        """
        analyze frequency of environmental terms in each text

        environmental_categories: dict of category names and related terms

        e.g., {'water': ['water', 'river', 'ocean', 'rain'],
               'agriculture': ['farm', 'crop', 'harvest', 'field']}
        """
        results = defaultdict(dict)
        # calculate frequency of each term in each category
        for doc_name, document in zip(self.names, self.documents):
            words = self.preprocess(document)
            total_words = len(words)

            # count frequency of each term in each category
            for category, terms in environmental_categories.items():
                count = sum(1 for word in words if word in terms)
                frequency = (count / total_words) * 1000  # per 1000 words
                results[doc_name][category] = frequency

        # return results as a dataFrame
        return pd.DataFrame(results).T
    def create_visualizations(self, env_data=None):
        """
        Create visualizations for each analysis component
        only used 2 for presentation
        """

        # 1. topic similarity Heatmap
        plt.figure(figsize=(18, 10))
        sns.heatmap(
            self.topic_similarities,
            cmap='YlOrRd',
            annot=False,
            fmt='.2f',
            square=True
        )
        plt.title('Topic Similarities Between Religious Texts', pad=20, size=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        # save as image
        plt.savefig('topic_similarities.png', bbox_inches='tight')

        plt.show()

        if env_data is not None:

            # 2. environmental term frequencies
            plt.figure(figsize=(20, 10))
            ax = env_data.plot(kind='bar', width=0.8)
            plt.title('Environmental Term Frequencies by Text', pad=20, size=16)
            plt.xlabel('Religious Texts', size=12)
            plt.ylabel('Frequency per 1000 words', size=12)
            plt.xticks(rotation=45, ha='right', size=6)
            plt.legend(title='Environmental Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # save as image
            plt.savefig('environmental_term_frequencies.png', bbox_inches='tight')

            plt.tight_layout()
            plt.show()

            # 3. environmental terms correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = env_data.T.corr()
            sns.heatmap(
                correlation_matrix,
                annot=False,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                vmin=-1, vmax=1
            )
            plt.title('Correlations Between Environmental Terms', pad=20, size=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0, size=12)
            plt.tight_layout()
            plt.show()

    def create_environmental_distribution(self, env_data):
        """
        create a distribution plot for environmental terms across texts
        """
        # melt the dataframe for easier plotting
        env_melted = env_data.reset_index().melt(
            id_vars=['Name_of_Text'],
            var_name='Environmental Category',
            value_name='Frequency'
        )

        # create box plots
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=env_melted,
            x='Environmental Category',
            y='Frequency',
            palette='Set3'
        )
        plt.title('Distribution of Environmental Terms Across Texts', pad=20, size=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Frequency per 1000 words')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def create_top_terms_plot(self, env_data, top_n=5):
        """
        create a plot showing top N texts for each environmental category
        """
        plt.figure(figsize=(15, 10))

        # number of categories
        n_categories = len(env_data.columns)
        n_rows = (n_categories + 1) // 2  # calculate number of rows needed

        for idx, category in enumerate(env_data.columns, 1):
            plt.subplot(n_rows, 2, idx)

            # sort values for this category
            top_texts = env_data[category].sort_values(ascending=False).head(top_n)

            # create horizontal bar plot
            bars = plt.barh(
                range(len(top_texts)),
                top_texts.values,
                color=plt.cm.Set3(idx / len(env_data.columns))
            )

            # add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}',
                    ha='left',
                    va='center',
                    fontweight='bold'
                )

            plt.title(f'Top {top_n} Texts - {category}')
            plt.yticks(range(len(top_texts)), top_texts.index, size=8)
            plt.xlabel('Frequency per 1000 words')

        plt.tight_layout()
        plt.show()


# define environmental terms to search for
environmental_categories = {
    'water': ['water', 'river', 'ocean', 'rain', 'sea', 'flood', 'stream', 'lake'],
    'agriculture': ['farm', 'crop', 'harvest', 'field', 'seed', 'plant', 'grow'],
    'climate': ['sun', 'wind', 'storm', 'heat', 'cold', 'season', 'weather'],
    'land': ['mountain', 'desert', 'forest', 'valley', 'hill', 'earth', 'soil'],
    'animals': ['cattle', 'sheep', 'bird', 'fish', 'beast', 'flock', 'herd']
}

# initialize the analyzer with the dataset
analyzer = ReligiousTextAnalyzer('Dataset_with_Text.csv')

# extract topics using LDA
topics = analyzer.extract_topics()

# calculate similarities between texts
similarities = analyzer.calculate_topic_similarities()

# analyze environmental terms
env_analysis = analyzer.analyze_environmental_terms(environmental_categories)

# create visualizations

analyzer.create_visualizations(env_analysis)
analyzer.create_environmental_distribution(env_analysis)
analyzer.create_top_terms_plot(env_analysis)

#Notes
""" see what topics were important for each religion outside of chosen topics """
#%%