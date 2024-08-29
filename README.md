# NanoGPT

In this repo, we realize a nano version of GPT from scratch. The model includes a plain Byte Pairing tokenizer and a decoder-only version of Transformer. We train the tokenizer with Shakespeare's books and then train the language model with five different books from Ernest Hemingway.

In the plain tokenizer, we tokenize the text by plain byte pairing algorithm. Although different corner cases are treated differently in tokenizers of for example GPT2, GPT3 and Llama, we keep our tokenizing process simple. 

The GPT model includes one embedding layer embed both data and position, followed by 6 layers of multi-head casual self-attenion blocks and a two layer Linear language model head for generation. The  hyperparameters can be found as below

| vocabulary size  | embedding dimension  | number of head |number of attention blocks |block size |
|---|---|---|---|---|
| 768| 384 |6|6 |64|



 A sample from the trained model is attached:



      “He can’t like Harris,” I said.
      Brett her eyes caught and for the great smoke out of him car. There are 
        their rollers. 

        Tt’s never fight behaving/ she said, ‘Marriest. You’ll be 
        undressing, Ettore.’ ^’ 

        ‘Yes you see?’ 

        ‘All right.’ 

        Weser brought the friend I translanted at the shany.’ 

        ‘We hall founded beer/ 

        ‘There will stop out try and not go up and dy. 
        Do they are. Xove We can.’ They stood up. They 
        were recountry. I took the moon and shook her hand. 
        I do not know how many relices. We were 
        back into the seat with them. I had exple to have 
        them that their rocks-case. We would Miss 
        tailardist. His so
        crew its of fruit that was like broken shining beside 
        with it. Any Americans who could get some 
        backles on good the beer, then she had 
        a waiter, ''Then’t even cone for triport of an very tinned-four^. 
        There were plenty of Austrian Englishman and the 
        stalled right in but I gave the trees in the early-card, then a dropped
        at the ambrait, some, steel, and the fish on the balcony,
        so the bed, going stroked into the grand out. It was crowd
        of killing the dark and the houses under the trees that looked was from the dancers of
        the firon shouths. The balcony came out onto the clothes. Those 
        were fine turned to me and from a village going into the
        table. Cohn shot down at Casablance. Live out of the big forest in it.
        When he hurt him I wake. They were on the others to serve
        about him six. He did not have a divce man, but he quaid not true
        known. He made no longer of this.

        Why nose was different and he knew that he would.  He was still lit and there he had
        not want them in the soil and take each rocks and years course hurried the swead
        hawks all through him and out with him.  He
        first went away.

        Just now the wind were noisy since he had bright.  The seatment of
        steadily.  The sun with a machanged los.  Perhaps from the shaft.  The shark
        was handing in the field.  Then the left hand-ut raw hat ct very worpost and the 
        main road to the grinner the post, sun-leafter the fields of an iron foot

The realization follows the instruction of Andrej Karpathy's lecture on GPT and tokenizer on Youtube:
https://www.youtube.com/@AndrejKarpathy