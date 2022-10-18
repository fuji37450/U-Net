if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade
kaggle datasets download -d saifkhichi96/bank-checks-signatures-segmentation-dataset
unzip bank-checks-signatures-segmentation-dataset.zip -d data/BCSD
rm bank-checks-signatures-segmentation-dataset.zip