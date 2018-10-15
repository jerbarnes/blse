import sys, os
import argparse
from blse import *


def main():
    pass

if __name__ == '__main__':
    main()
    """
    Search for the best hyperparameters for the blse model. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--langs', 
                        help="target language: es, ca, eu", 
                        nargs='+', default=['es','ca','eu'])
    parser.add_argument('-e', '--epochs', type=int, 
                        default=200)
    parser.add_argument('-a', '--alphas',
                        help="list of alphas for hyperparameter search (default: [0.1 - 0.9])",
                        nargs='+',
                        type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('-bs', '--batch_sizes', 
                        help="batch sizes for hyperparameter search (default: [20, 50, 80, 100, 200]",
                        nargs='+', 
                        type=int, 
                        default=[20, 50, 80, 100, 200])
    parser.add_argument('-src_vecs', 
                        help="source embedding file (default: embeddings/original/google.txt)",
                        default='embeddings/original/google.txt')
    parser.add_argument('-trg_vecs_dir', 
                        help='directory of target language vectors. Vectors should be named sg-300-LANG-.txt, where LANG is the two letter language identifier, i.e. ES for Spanish', 
                        default='embeddings/original/sg-300-es.txt')
    parser.add_argument('-lexicon_dir', 
                        help='translation lexicon directory (default: lexicons/bingliu)', 
                        default='lexicons/bingliu')

    args = parser.parse_args()

    for lang in args.langs:

        # Import monolingual vectors
        print('importing word embeddings')
        src_vecs = WordVecs(args.src_vecs)
        trg_vec_file = os.path.join(args.trg_vecs_dir, 'sg-300-{0}.txt'.format(lang))
        trg_vecs = WordVecs(trg_vec_file)

        for bi in [True, False]:
            
            # import datasets (representation will depend on final classifier)
            print('importing datasets')
    
            dataset = General_Dataset(os.path.join('datasets', 'en', 'opener_sents'), None,
                                  binary=bi, rep=words, one_hot=False)
    
            cross_dataset = General_Dataset(os.path.join('datasets', lang, 'opener_sents'), None,
                                  binary=bi, rep=words, one_hot=False)


            # Get sentiment synonyms and antonyms to check how they move during training
            synonyms1, synonyms2, neg = get_syn_ant('en', src_vecs)
            cross_syn1, cross_syn2, cross_neg = get_syn_ant(lang, trg_vecs)

            # Import translation pairs
            trans = os.path.join(args.lexicon_dir, 'en-{0}.txt'.format(lang))
            pdataset = ProjectionDataset(trans, src_vecs, trg_vecs)


            for batch_size in args.batch_sizes:

                for alpha in args.alphas:

                    # initialize classifier
                    if bi:
                        blse = BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                                    output_dim=2,
                                    src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                                    trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg,
                                    )
                    else:
                        blse = BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                                    output_dim=4,
                                    src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                                    trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg,
                                    )

                    # train model
                    print('training model')
                    print('Parameters:')
                    print('lang:       {0}'.format(lang))
                    print('binary:     {0}'.format(bi))
                    print('epoch:      {0}'.format(args.epochs))
                    print('alpha:      {0}'.format(alpha))
                    print('batchsize:  {0}'.format(batch_size))
                    print('src vecs:   {0}'.format(args.src_vecs))
                    print('trg_vecs:   {0}'.format(trg_vec_file))
                    print('trans dict: {0}'.format(trans))
                    if bi:
                        b = 'bi'
                    else:
                        b = '4cls'

                    # Save model weights here
                    weight_dir = os.path.join('models', 'blse', '{0}-{1}-{2}'.format('opener_sents', lang, b))
                    os.makedirs(weight_dir, exist_ok=True)
                    print(weight_dir)

                    blse.fit(pdataset._Xtrain, pdataset._ytrain,
                            dataset._Xtrain, dataset._ytrain,
                            weight_dir=weight_dir,
                            alpha=alpha, epochs=args.epochs,
                            batch_size=batch_size)

                    # get the best weights
                    best_f1, best_params, best_weights = get_best_run(weight_dir)
                    epochs, batch_size, alpha = best_params
                    blse.load_weights(best_weights)
                    
                    # evaluate
                    if bi:
                        # ble.plot(outfile=os.path.join('figures', 'syn-ant', lang, 'ble', '{0}-bi-alpha{1}-epoch{2}-batch{3}.pdf'.format(args.dataset, alpha, epochs, batch_size)))
                        blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False)
                        blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False,
                                     outfile=os.path.join('predictions', lang, 'blse', 'opener_sents-bi-alpha{0}-epoch{1}-batch{2}.txt'.format(alpha, epochs, batch_size)))
                    else:
                        # ble.plot(outfile=os.path.join('figures', 'syn-ant', lang, 'ble', '{0}-4cls-alpha{1}-epoch{2}-batch{3}.pdf'.format(args.dataset, alpha, epochs, batch_size)))
                        blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False)
                        blse.evaluate(cross_dataset._Xtest, cross_dataset._ytest, src=False,
                                     outfile=os.path.join('predictions', lang, 'blse', 'opener_sents-4cls-alpha{0}-epoch{1}-batch{2}.txt'.format(alpha, epochs, batch_size)))


