export HF_HOME=/ocean/projects/ees250005p/tho2/.cache
mkdir /local/data
curl https://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip --output /local/data/train_img.zip
curl https://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip --output /local/data/val_img.zip
curl https://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip --output /local/data/test1_img.zip
echo "Download complete. Unzipping files..."
unzip /local/data/train_img.zip -d /local/data/
unzip /local/data/val_img.zip -d /local/data/
unzip /local/data/test1_img.zip -d /local/data/
echo "Unzipping complete."