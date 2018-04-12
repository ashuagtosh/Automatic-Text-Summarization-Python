import nltk

from nltk.tokenize import sent_tokenize,word_tokenize

example_text = "Born and brought up in Delhi, he was a brilliant student and an all rounder. At St. Columba's School he won the prestigious 'Sword of Honour'. He was also the captain of his football, hockey and cricket teams all at the same time. After sustaining a football injury, he started to consider acting as a career. He graduated with honours in Economics from Hansraj College and did his masters in mass communications from Jamiya Miliya Islamia. Khan then went on to study theatre under the reputed Mr. Barry John whom he credits for carving out the actor inside him but his mentor considers his pupil as being solely responsible for his greatness and glory citing Shahrukh to have the perfect combination of 'talent, perseverance, belief in yourself, a never-give-up-attitude and of course, a bit of luck'; factors needed for success."
##print(sent_tokenize(example_text))
##print(word_tokenize(example_text))
for i in word_tokenize(example_text):
    print(i)
