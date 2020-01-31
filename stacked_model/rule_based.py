#!/usr/bin/env python3

from sklearn.metrics import classification_report
from collections import Counter, defaultdict
import json 
# TODO:
# Different sentiment calculations !!!
# Add/Eliminate words to the different sets !!!
# Not sure how to handle neutral, think about this
# Might be better to just copy paste the neg + pos word sets, so we are not reliant on internet and a specific site
# Try different thresholds

def get_word_list(url):
    '''
    Get positive and negative word sets from:
        - https://www.enchantedlearning.com/wordlist/positivewords.shtml
        - https://www.enchantedlearning.com/wordlist/negativewords.shtml
    '''
    try:
        from bs4 import BeautifulSoup
        import requests
        content = requests.get(url).text
        soup = BeautifulSoup(content, 'html.parser')
        return set(item.get_text(strip=True) for item in soup.find_all(class_="wordlist-item"))
    except Exception as e:
        print(f"Sadly, the URL does not seem to work, error={e}")


def sentiment_calculation(pos, neg, neu, threshold=4):
    '''
    Return overall sentiment
    '''
    sentiment = False
    pred_dict = {}
    pred_dict[pos] = "positive"
    pred_dict[neg] = "negative"
    pred_dict[neu] = "neutral"
    prediction = max(pos, neg, neu)
    if prediction >= threshold:
        sentiment = pred_dict[prediction]
    return sentiment


def rule_based_predictor(X, y, y_og, inspect=False):
    '''
    Overwrite cetain old predictions with new ones according to rules set by sentiment_calculation()
    '''
    new_y, replaced_ids = [], []

    positive_set = get_word_list("https://www.enchantedlearning.com/wordlist/positivewords.shtml") | {"magnificent", 'celebrado','poderoso','aventuras','fabuloso','simple','ligero','radiante','uno','soleado','sorprendente','Fresco','significativo','arriba','instintivo','abierto','aclamado','pulido','estupendo','felicidad','cien por ciento','legendario','fácil','afortunado','multa','conmovedor','complaciente','instantáneo','enérgico','endosado','hábil','aprobar','amistoso','recompensa','agradable','genio','feliz','creciente','productivo','especial','adivinar','emocionante','hermoso','espumoso','deleite','maravilloso','Guau','animado','beneficioso','victorioso','estimado','distinguido','curación','rápido','positivo','resonando','familiar','motivando','espiritual','creer','eficiente','popular','afilado','fenomenal','refinado','prominente','agradable','vibrante','experto','afluente','alentador','inventivo','excelente','atractivo','sabroso','restaurada','realizar','fácil','justa','encantador','divertido','gratificante','listo','gracioso','refrescante','valiente','clásico','agradable','logro','intelectual','bien','maravilloso','impresionante','absolutamente','jubiloso','Bienvenido','adecuado','honesto','vivaz','acción','recto','secundario','maestro','de confianza','limpiar','de principios','seguro','emocionante','logro','generoso','victoria','milagroso','aptitud','lúcido','elegante','encantador','fantástico','Excelente','hermosa','nutritivo','Perfecto','virtuoso','verde','genuino','luminoso','constante','notable','entusiasta','imagina','experto','agraciado','vigoroso','paraíso','completo','alegrarse','calma','vertical','celestial','bueno','extático','efervescente','optimista','idea','optimista','de acuerdo','confiando','transformadora','bonito','irreal','honorable','veraz','jovial','Listo','morder','abundante','atractivo','sonrisa','ahora','maravilloso','encantador','mérito','estupendo','admirar','alegría','intuitivo','Bravo','saludable','vital','conocimiento','creativo','orgulloso','enérgico','respetado','angelical','decoroso','ético','elección','abundante','natural','celo','abundante','armonioso','instante','afirmativo','alegre','seguro','tipo','adorable','protegido','dando','satisfactorio','digno','burbujeante','inteligente','honrado','increíble','celoso','electrizante','risa','tapas','tranquilo','campeón','ideal','emocionante','sí','innovador','generosidad','todo','brillante','tranquilizador','gratis','meritorio','serio','esencial','preparado','linda','éxito','maravilloso','brillante','imaginativo','valiente','felicitación','rico','magnífico','próspera','elogiar','crianza','transformadora','novela','famoso','valorado','exitoso','calidad','tranquilo','atractivo','súper','bueno','floreciente','abrazo','guay','saludable','suerte','atractivo','encantador','favorable','Exquisito','asombroso','bonita','aceptado','activo','maravilloso','inquebrantable','atractivo','deslumbrante','mueca','eficaz','innovar','Progreso','clásico','energizado','emocionante','Derecha','independiente','compuesto','robusto','cierto','soberbio','aprendido'} 
    negative_set = get_word_list("https://www.enchantedlearning.com/wordlist/negativewords.shtml") | {"turd"}
    neutral_set = {"implementation"}

    for idx, ((words, *_), label) in enumerate(zip(X, y)):
        word_set = set([word.lower() for word in words])
        pos = len(word_set & positive_set)
        neg = len(word_set & negative_set)
        neu = len(word_set & neutral_set)
        sentiment = sentiment_calculation(pos, neg, neu)
        if sentiment:
            replaced_ids.append(idx)
            new_y.append(sentiment)
        else:
            new_y.append(label)
    print(f"The rule based system, replaced {len(replaced_ids)} predictions!")            
    if inspect:
        print(f"The following ids: {replaced_ids} were replaced by the rule based predictor!\n")
        print(classification_report([new_y[i] for i in replaced_ids], [y_og[i] for i in replaced_ids], digits=2))
        print()
    return new_y


def emoji_based_predictor_los(X):
    import re
    import emoji
    import requests
    from bs4 import BeautifulSoup
    from collections import defaultdict
    
    predictions = {}
    smiley_set = set()
    smiley_score_dict = defaultdict(dict)

    for line in X:
        sentence = " ".join(line)
        for smiley in re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', sentence):
            smiley_set.add(smiley)
        emojis = [chr for chr in sentence if chr in emoji.UNICODE_EMOJI]
        for emo in emojis:
            smiley_set.add(emo)
    
    url = 'http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table = soup.find("table")

    for row in table.find_all("tr"):
        headers = table.find_all("th")
        for idx, header in enumerate(headers):
            if header.text == "Char":
                emoji_char = idx
            if header.text == "Neg[0...1]":
                neg_score = idx
            if header.text == "Neut[0...1]":
                neu_score = idx
            if header.text == "Pos[0...1]":
                pos_score = idx                
            if header.text == "Sentiment score[-1...+1]":
                avg_score = idx                
        values = row.find_all("td")
        try:
            smiley_score_dict[values[emoji_char].text]["neg"] = values[neg_score].text
            smiley_score_dict[values[emoji_char].text]["neu"] = values[neu_score].text
            smiley_score_dict[values[emoji_char].text]["pos"] = values[pos_score].text
            smiley_score_dict[values[emoji_char].text]["avg"] = values[avg_score].text
        except IndexError as e:
            continue

    # Manually annotated smileys
    smiley_score_dict[":)"]["avg"] = 0.3
    smiley_score_dict[":D"]["avg"] = 0.4
    smiley_score_dict[":P"]["avg"] = 0.5
    smiley_score_dict[":x"]["avg"] = 0.1
    smiley_score_dict[":X"]["avg"] = 0.1
    smiley_score_dict[":("]["avg"] = -0.3
    smiley_score_dict["=)"]["avg"] = 0.3
    smiley_score_dict["=D"]["avg"] = 0.4
    smiley_score_dict["=P"]["avg"] = 0.5
    smiley_score_dict["=x"]["avg"] = 0.1
    smiley_score_dict["=X"]["avg"] = 0.1
    smiley_score_dict["=("]["avg"] = -0.3
    smiley_score_dict[";)"]["avg"] = 0.5
    smiley_score_dict[";("]["avg"] = -0.3
    smiley_score_dict[":^)"]["avg"] = 0.2
    smiley_score_dict[":-)"]["avg"] = 0.3
    smiley_score_dict[":-("]["avg"] = -0.3
    smiley_score_dict[":-D"]["avg"] = 0.4
    smiley_score_dict["🍾"]["avg"] = 0.2
    smiley_score_dict["🤦"]["avg"] = -0.2
    smiley_score_dict["🤷"]["avg"] = 0
    smiley_score_dict["🏅"]["avg"] = 0.7
    smiley_score_dict["🌮"]["avg"] = 0.6
    smiley_score_dict["🤑"]["avg"] = 0.2
    smiley_score_dict["🤓"]["avg"] = 0.2
    smiley_score_dict["🤣"]["avg"] = 0.4
    smiley_score_dict["🤒"]["avg"] = -0.4
    smiley_score_dict["🤢"]["avg"] = -0.5
    smiley_score_dict["🤤"]["avg"] = 0.8
    smiley_score_dict["🤥"]["avg"] = -0.3
    smiley_score_dict["🤐"]["avg"] = 0.1
    smiley_score_dict["🙁"]["avg"] = -0.3
    smiley_score_dict["🕺"]["avg"] = 0.4
    smiley_score_dict["🤶"]["avg"] = 0.3
    smiley_score_dict["🤞"]["avg"] = 0.3
    smiley_score_dict["🖖"]["avg"] = 0.5
    smiley_score_dict["🥑"]["avg"] = 0.2
    smiley_score_dict["🤗"]["avg"] = 0.8
    smiley_score_dict["🙃"]["avg"] = 0.1
    smiley_score_dict["™"]["avg"] = -0.2
    smiley_score_dict["🖕"]["avg"] = -0.8
    smiley_score_dict["🤕"]["avg"] = -0.4
    smiley_score_dict["🤔"]["avg"] = -0.15
    smiley_score_dict["🥂"]["avg"] = 0.2
    smiley_score_dict["🤧"]["avg"] = -0.2
    smiley_score_dict["🙄"]["avg"] = -0.3
    smiley_score_dict["🤘"]["avg"] = 0.3
    smiley_score_dict["🙂"]["avg"] = 0.3
    smiley_score_dict["🤙"]["avg"] = 0.2

    neg_c = 0
    pos_c = 0
    neu_c = 0
    index_counter = 0
    
    for line in X:
        sentiment = 0
        count = 0
        sentence = " ".join(line)
        for smiley in re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', sentence):
            try:
                sentiment += smiley_score_dict[smiley]["avg"]
                count += 1
            except:
                continue
        emojis = [chr for chr in sentence if chr in emoji.UNICODE_EMOJI]
        for emo in emojis:
            try:
                sentiment += smiley_score_dict[emo]["avg"]
                count += 1
            except:
                continue
        if sentiment >= .7:
            predictions[sentence] = "positive"
            pos_c += 1
            
        elif sentiment <= -.31:
            predictions[sentence] = "negative"
            neg_c += 1
            
        elif sentiment != 0:
            if count == 0:
                count = 1
            if ((sentiment/count) <= 0.1 and (sentiment/count) >= -0.1):
                predictions[sentence] = "neutral"
                neu_c += 1
            else:
                continue

        else:
            continue

    print(f"Emojis: Neg ({neg_c}), Neu ({neu_c}), Pos ({pos_c})")
    return predictions


def rule_based_predictor_los(X, translated=False, emoji=False):
    '''
    Overwrite cetain old predictions with new ones according to rules set by sentiment_calculation()
    '''

    positive_set = get_word_list("https://www.enchantedlearning.com/wordlist/positivewords.shtml") |{"magnificent"}
    negative_set = get_word_list("https://www.enchantedlearning.com/wordlist/negativewords.shtml") |{"turd"}
    neutral_set = {"implementation", "neutral", "yoghurt"}

    with open("../data/negative_wordlist.txt", "r") as f:
        for line in f:
            line = line.strip()
            negative_set.add(line)

    with open("../data/positive_wordlist.txt", "r") as f:
        for line in f:
            line = line.strip()
            positive_set.add(line)

    if translated:
        positive_set = positive_set | {"magnificent", 'celebrado','poderoso','aventuras','fabuloso','simple','ligero','radiante','uno','soleado','sorprendente','Fresco','significativo','arriba','instintivo','abierto','aclamado','pulido','estupendo','felicidad','cien por ciento','legendario','fácil','afortunado','multa','conmovedor','complaciente','instantáneo','enérgico','endosado','hábil','aprobar','amistoso','recompensa','agradable','genio','feliz','creciente','productivo','especial','adivinar','emocionante','hermoso','espumoso','deleite','maravilloso','Guau','animado','beneficioso','victorioso','estimado','distinguido','curación','rápido','positivo','resonando','familiar','motivando','espiritual','creer','eficiente','popular','afilado','fenomenal','refinado','prominente','agradable','vibrante','experto','afluente','alentador','inventivo','excelente','atractivo','sabroso','restaurada','realizar','fácil','justa','encantador','divertido','gratificante','listo','gracioso','refrescante','valiente','clásico','agradable','logro','intelectual','bien','maravilloso','impresionante','absolutamente','jubiloso','Bienvenido','adecuado','honesto','vivaz','acción','recto','secundario','maestro','de confianza','limpiar','de principios','seguro','emocionante','logro','generoso','victoria','milagroso','aptitud','lúcido','elegante','encantador','fantástico','Excelente','hermosa','nutritivo','Perfecto','virtuoso','verde','genuino','luminoso','constante','notable','entusiasta','imagina','experto','agraciado','vigoroso','paraíso','completo','alegrarse','calma','vertical','celestial','bueno','extático','efervescente','optimista','idea','optimista','de acuerdo','confiando','transformadora','bonito','irreal','honorable','veraz','jovial','Listo','morder','abundante','atractivo','sonrisa','ahora','maravilloso','encantador','mérito','estupendo','admirar','alegría','intuitivo','Bravo','saludable','vital','conocimiento','creativo','orgulloso','enérgico','respetado','angelical','decoroso','ético','elección','abundante','natural','celo','abundante','armonioso','instante','afirmativo','alegre','seguro','tipo','adorable','protegido','dando','satisfactorio','digno','burbujeante','inteligente','honrado','increíble','celoso','electrizante','risa','tapas','tranquilo','campeón','ideal','emocionante','sí','innovador','generosidad','todo','brillante','tranquilizador','gratis','meritorio','serio','esencial','preparado','linda','éxito','maravilloso','brillante','imaginativo','valiente','felicitación','rico','magnífico','próspera','elogiar','crianza','transformadora','novela','famoso','valorado','exitoso','calidad','tranquilo','atractivo','súper','bueno','floreciente','abrazo','guay','saludable','suerte','atractivo','encantador','favorable','Exquisito','asombroso','bonita','aceptado','activo','maravilloso','inquebrantable','atractivo','deslumbrante','mueca','eficaz','innovar','Progreso','clásico','energizado','emocionante','Derecha','independiente','compuesto','robusto','cierto','soberbio','aprendido'} 
        negative_set = negative_set | {"turd"}

        with open("../data/positive_wordlist_spanish.txt", "r") as f:
            for line in f:
                line = line.strip()
                positive_set.add(line)
        
        with open("../data/negative_wordlist_spanish.txt", "r") as f:
            for line in f:
                line = line.strip()
                negative_set.add(line)

    predictions = {}
    for idx, words in enumerate(X):
        sentence = [word.lower() for word in words]
        word_count = Counter(sentence)
        word_set = set(sentence)
        pos = sum(word_count[i] for i in (word_set & positive_set))
        neg = sum(word_count[i] for i in (word_set & negative_set))
        neu = sum(word_count[i] for i in (word_set & neutral_set))
        sentiment = sentiment_calculation(pos, neg, neu)
        if sentiment:
            predictions[" ".join(words)] = sentiment
        else:
            continue
    if emoji:
        emoji_predictions = emoji_based_predictor_los(X)
        predictions = {**predictions, **emoji_predictions} 
        print(f"The rule based system, made {len(predictions)} (emoji={len(emoji_predictions)}({round((len(emoji_predictions)/len(predictions))*100, 2)}%), words={len(predictions)-len(emoji_predictions)}({round(100-(len(emoji_predictions)/len(predictions))*100, 2)}%)) predictions!")            
    else:
        print(f"The rule based system, made {len(predictions)} predictions!")            
    return predictions


def main():
    extra_data_switch = False
    if extra_data_switch:
        extra_data = []
        with open("../data/extra_spanglish_tweets_gosse.txt", "r") as f:
            for line in f:
                words = line.strip().split()
                extra_data.append(words)

        rb_predictions, _ = rule_based_predictor_los(extra_data, translated=True, emoji=True)
        with open("../data/extra_data_labelled.txt", "w") as f:
            for k, v in rb_predictions.items():
                f.write(k + "\t" + v + "\n")
    else:
        extra_data = []
        with open('../translator/EN_file.json') as injson:
            en_file = json.load(injson)
        with open('../translator/ES_file.json') as injson:
            es_file = json.load(injson)
        with open('../normalisation/normalised_extra.json', 'r') as infile:
            extra_data_json = json.load(infile)
        for item in extra_data_json:
            extra_data.append(" ".join(i[0] for i in item).split())
        rb_predictions = rule_based_predictor_los(extra_data, translated=True, emoji=True)
        print(len(en_file))
        print(len(es_file))
        with open("../data/extra_data_normalised_labelled.txt", "w") as f:
            for k, v in rb_predictions.items():
                f.write(k + "\t" + v + "\n")
        new_en_file, new_es_file = [], []
        index = 0
        index_dict = {}
        for item in extra_data:
            if " ".join(item) in rb_predictions.keys():
                index += 1
                index_dict[" ".join(item)] = index
        for item in rb_predictions.keys():
            new_en_file.append(en_file[index_dict[item]])
            new_es_file.append(es_file[index_dict[item]])
        print(len(new_en_file))
        print(len(new_es_file))
        with open('../data/extra_data_normalised_spanish.txt', 'w') as outfile:
            for item in new_es_file:
                outfile.write(f"{item}\n")

        with open('../data/extra_data_normalised_english.txt', 'w') as outfile:
            for item in new_en_file:
                outfile.write(f"{item}\n")

if __name__ == "__main__":
    main()
