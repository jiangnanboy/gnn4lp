import pandas as pd

def cora_content_onehot_class_process(cora_content_file, cora_content_class_file):
    cora_feature_names = [f"w{i}" for i in range(1433)]
    cora_raw_content = pd.read_csv(cora_content_file,
                                   sep='\t',
                                   header=None,
                                   names=['id', *cora_feature_names, 'subject'])
    cora_content_onehot_subject = pd.get_dummies(cora_raw_content, columns=['subject'])
    cora_content_onehot_subject.to_csv(cora_content_class_file, sep=' ', header=False, index=False)

    print('cora_content_onehot_subject: {}'.format(cora_content_onehot_subject.shape))
