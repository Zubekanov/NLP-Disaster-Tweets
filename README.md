# NLP Disaster Tweets - Practice Competition Notes

> Although this is only a practice competition, Kaggle requires agreeing to competition terms to access the dataset, so it is not committed here out of consideration.

## Running the Code

The code can be run by inserting the relevant `.csv` files into the `data/` directory.

---

## Experiment Logs

### First Attempt
**The model way overfits the data.**

![graphs](https://github.com/user-attachments/assets/0a195ffd-d1f0-4494-b3a8-3290d42e913f)

---

### Second Attempt
- Reduced `epochs` to 50.
- Limited `vectorizer max_features` to 1000.

**Same overfitting problem persists.**

![graph2](https://github.com/user-attachments/assets/74d5fe44-2009-40d1-8d67-7a2be27e4b22)

---

### Third Attempt
- Simplified network to a single dense layer of 64 neurons.

**Validation loss initially drops, but rises too early and remains too high.**

![graph3](https://github.com/user-attachments/assets/6bbe73f7-f6a4-400e-ac19-6f7af836333f)

---

### Fourth Attempt
- Dropped `location` from categorical features, as it was very noisy  
  (e.g., "Happily Married with 2 kids", "VISIT MY YOUTUBE CHANNEL.")  

**Validation loss reaches its minimum later, but still has the same minimum value.**

![graph4](https://github.com/user-attachments/assets/4f5f8599-ee13-4eaa-ad8e-a1c9c44bd4d7)

## Final Result

Final accuracy was 80%, which I accepted for this level of training.
