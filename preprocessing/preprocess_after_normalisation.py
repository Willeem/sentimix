import json 
from collections import defaultdict
def replace_lol(filename):
    output_list = []
    for item in filename:
        output_dict = {'text':[],'sentiment':""}
        output_dict['sentiment'] = item['sentiment']
        for i in item['text']:
            a = i[0].replace('laughing out loud','lol')
            output_dict['text'].append([a,i[1]])
        output_list.append(output_dict)
            
    return output_list
def join_urls(filename):
    output_list = []
    for item in filename:
        output_dict = {'text':[],'sentiment':""}
        output_dict['sentiment'] = item['sentiment']
        for l in item['text']:
            if l[0] == 'https':
                a = l[0] + "".join(i[0] for i in item['text'][item['text'].index(l)+1:])
                output_dict['text'].append([a,'other'])
                break
            else:
                output_dict['text'].append(l)
        output_list.append(output_dict)
    return output_list 
def main():
    with open('normalisation/normalised_spanglish.json','r') as infile:
        normalised_spanglish = json.load(infile)

    with open('normalisation/normalised_hindi_train.json','r') as infile:
        normalised_hindi_train = json.load(infile)

    with open('normalisation/normalised_hindi_trial.json','r') as infile:
        normalised_hindi_trial = json.load(infile)
    # with open('normalisation/normalised_hingtrain_without_lol.json','r') as infile:
    #     normalised_hindi_trial = json.load(infile)   
    
    spanglish_postprocessed = replace_lol(normalised_spanglish)
    hingtrain_without_lol = replace_lol(normalised_hindi_train)
    hingtrial_without_lol = replace_lol(normalised_hindi_trial)
    hingtrain_postprocessed = join_urls(hingtrain_without_lol)
    hingtrial_postprocessed = join_urls(hingtrial_without_lol)
    # replace_lol(normalised_hindi_trial)
    
    with open('normalisation/normalised_spanglish_postprocessed.json', 'w') as outfile:
        json.dump(spanglish_postprocessed,outfile)
    with open('normalisation/normalised_hingtrain_postprocessed.json', 'w') as outfile:
        json.dump(hingtrain_postprocessed,outfile)
    with open('normalisation/normalised_hingtrial_postprocessed.json', 'w') as outfile:
        json.dump(hingtrial_postprocessed,outfile)

if __name__ == "__main__":
    main()
