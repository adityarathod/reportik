import os

from loader import DataManager
from model import NewsSummarizationModel

example_text = '''Police and ambulances raced to a shooting at a food festival in California on Sunday, and video posted on social media showed people at the event running for cover as shots rang out.

Few details were immediately available, but a police spokesman said there were casualties.

NBC Bay Area reported that ambulance crews were told 11 people were “down” after the shooting on the last day of the Gilroy Garlic Festival, an annual three-day event south of San Jose.

Footage uploaded to social media appeared to show festival attendees scattering in confusion as loud popping sounds could be heard in the background.

“What’s going on?” a woman can be heard asking on one video. “Who’d shoot up a garlic festival?”

Evenny Reyes, 13, told the San Jose Mercury News that at first she thought the sound of gunfire was fireworks. But then she saw someone with a wounded leg.

“We were just leaving and we saw a guy with a bandana wrapped around his leg because he got shot,” Reyes told the newspaper. “There was a little kid hurt on the ground. People were throwing tables and cutting fences to get out.”

Another witness, Maximo Rocha, a volunteer with the Gilroy Browns youth football team, said he saw many people on the ground, but he could not be sure how many may have been shot and how many were trying to protect themselves.

He told NBC Bay Area that “quite a few” were injured, “because I helped a few.”

Founded in 1979, the Gilroy Garlic Festival features food, drink, live entertainment and cooking competitions. It says it is hosted by volunteers and describes itself as the world’s greatest summer food festival.

It was being held at the outdoor Christmas Hill Park, where weapons of any kind are prohibited, according to the event’s website.

To provide a safe, family-friendly atmosphere, it said, entry was refused to anyone wearing clothing or paraphernalia indicating membership in a gang, including a motorcycle club.

Festival officials did not immediately respond to requests for comment.

Gilroy is about 30 miles (48 km) southeast of San Jose.'''


def main():
    latent_dim = 64
    embedding_size = 100
    num_cells = 8
    manager = DataManager(saved_dir='./data', embedding_size=embedding_size)
    model = NewsSummarizationModel(manager)
    model.build_model(latent_dim=latent_dim, depth=num_cells)
    print('[INFO] Model sanity check:')
    model.model.summary()
    # model.load('./cnbc-seq2seq-attn-weights.h5')
    # model.load('./cnbc-64-epoch-2-attn-weights.h5')
    for i in range(3):
        j = i + 1
        print(f'[INFO] Initiating epoch {j}...')
        model.train(epochs=1)
        model.save(
            os.getcwd(),
            f'cnbc-{latent_dim}-units-{embedding_size}-emb-epoch-{j}.h5'
        )
        print(model.evaluate())
        print(model.infer(example_text))


if __name__ == '__main__':
    main()
