import argparse
import sys
from vw import VW

__author__ = 'grf'


def main():
    parser = argparse.ArgumentParser()

    #parse vw
    parser.add_argument('--vw', type=str, help='Location to vw executable')

    #parse source
    parser.add_argument('-d', '--data', type=str, help='Example Set')
    parser.add_argument('-c', '--cache', action='store_true', help='Use a cache.  The default is data.cache')
    parser.add_argument('--cache_file', type=str, default='data.cache', help='The location of the cache file')
    parser.add_argument('-k', '--kill_cache', action='store_true',
                        help='Do not reuse existing cache: create a new one always')

    #parse feature tweaks
    parser.add_argument('--hash', type=str, default='all',
                        help='How to hash the features. Available options: strings, all')
    parser.add_argument('--ignore', type=str, help='Ignore namespaces beginning with character <arg>')
    parser.add_argument('--keep', type=str, help='Keep namespaces beginning with character <arg>')
    parser.add_argument('-b', '--bit_precision', type=int, default=29, help='Number of bits in the feature table')
    parser.add_argument('--noconstant', action='store_true', help='Do not add a constant feature')
    parser.add_argument('-C', '--constant', type=float, help='Set initial value of constant')
    #parser.add_argument('--ngram')
    #parser.add_argument('--skips')
    #parser.add_argument('--affix')
    #parser.add_argument('--spelling')
    #parser.add_argument('-q', '--quadratic')
    #parser.add_argument('-q:')
    #parser.add_argument('--cubic')

    #parse example tweaks
    parser.add_argument('-t', '--testonly', type=str, help='Ignore label information and just test')
    parser.add_argument('--holdout_off', action='store_true', help='No holdout data in multiple passes')
    parser.add_argument('--holdout_period', type=int, help='Holdout period for test only')
    parser.add_argument('--holdout_after', type=int,
                        help='Holdout after n training examples, default off (disables holdout_period)')
    parser.add_argument('--termination', type=float, help='Termination threshold')
    parser.add_argument('--early_terminate', type=int, default=3,
                        help='Specify the number of passes tolerated when holdout loss doesn\'t decrease before early termination, default is 3')
    parser.add_argument('--passes', type=int, default=1, help='Number of Training passes. Default is 1')
    #parser.add_argument('--initial_pass_length')
    #parser.add_argument('--examples')
    #parser.add_argument('--min_prediction')
    #parser.add_argument('--max_prediction')
    parser.add_argument('--sort_features', action='store_true',
                        help='Turn this on to disregard order in which features have been defined. This will lead to smaller cache sizes')
    parser.add_argument('--loss_function', type=str, default='logistic',
                        help='Specify the loss function to be used, uses squared by default. Currently available ones are squared, classic, hinge, logistic and quantile. Default is logistic')
    #parser.add_argument('--quantile_tau')
    parser.add_argument('--l1', type=float, default=7., help='l1 lambda. Default is 7')
    parser.add_argument('--l2', type=float, default=0., help='l1 lambda. Default is 0')
    parser.add_argument('--ftrl_alpha', type=float, help='Learning rate for ftrl-proximal optimization')
    parser.add_argument('--ftrl_beta', type=float, help='FTRL beta')
    parser.add_argument('--progressive_validation', type=str, default='ftrl.evl',
                        help='File to record progressive validation for ftrl-proximal')

    #parse output predictions
    parser.add_argument('-p', '--predictions', type=str, help='File to output predictions to')
    #parser.add_argument('-r', '--raw_predictions')

    #parse output model
    parser.add_argument('-f', '--final_regressor', type=str, help='Final regressor')
    parser.add_argument('--readable_mode', type=str, help='Output human-readable final regressor with numeric features')
    parser.add_argument('--invert_hash', type=str,
                        help='Output human-readable final regressor with feature names.  Computationally expensive')
    #parser.add_argument('--save_resume')
    #parser.add_argument('--save_per_pass')
    #parser.add_argument('--output_feature_regularizer_binary')
    #parser.add_argument('--output_feature_regularizer_text')

    #parse base algorithm
    parser.add_argument('--sgd', action='store_true', help='Use regular stochastic gradient descent update')
    parser.add_argument('--ftrl', action='store_true', help='Use ftrl-proximal optimization')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive, individual learning rates')
    parser.add_argument('--invariant', action='store_true', help='Use safe/importance aware updates')
    parser.add_argument('--normalized', action='store_true', help='Use per feature normalized updates')
    parser.add_argument('--exact_adaptive_norm', action='store_true',
                        help='Use current default invariant normalized adaptive update rule')
    parser.add_argument('--bfgs', action='store_true', help='Use bfgs optimization')
    parser.add_argument('--lda', type=int, help='Run lda with <int> topics')
    parser.add_argument('--rank', type=int, help='Rank for matrix factorization')
    parser.add_argument('--noop', action='store_true', help='Do no learning')
    parser.add_argument('--print', action='store_true', help='Print examples')
    parser.add_argument('--ksvm', action='store_true', help='Kernel svm')
    #parser.add_argument('--sendto')

    #parse scorer reductions
    parser.add_argument('--nn', type=int, help='Use sigmoidal feedforward network with <k> hidden units')
    parser.add_argument('--new_mf', help='Use new, reduction-based matrix factorization')
    parser.add_argument('--autolink', type=int, help='Create link function with polynomial d')
    parser.add_argument('--lrq', type=str, help='Use low rank quadratic features')
    #parser.add_argument('--lrqdropout')
    #parser.add_argument('--stage_poly')
    #parser.add_argument('--active')

    #parse score users
    #parser.add_argument('--top')
    #parser.add_argument('--binary')
    #parser.add_argument('--oaa')
    #parser.add_argument('--ect')
    #parser.add_argument('--log_multi')
    #parser.add_argument('--csoaa')
    #parser.add_argument('--csoaa_ldf')
    #parser.add_argument('--wap_ldf')

    #parse contextual bandit options
    #parser.add_argument('--cb')
    #parser.add_argument('--cbify')

    #parse search
    parser.add_argument('--search', type=int,
                        help='Use search-based structured prediction, argument=maximum action id or 0 for LDF"')

    #parse VW options
    parser.add_argument('--random_seed', type=int, help='Seed random number generator')
    #parser.add_argument('ring_size')
    parser.add_argument('-l', '--learning_rate', type=float, help='Set learning rate')
    parser.add_argument('--power_t', type=float, help='t power value')
    parser.add_argument('--decay_learning_rate', type=float, help='Set decay factor for learning_rate between passes')
    parser.add_argument('--initial_t', type=float, help='Initial t value')
    parser.add_argument('--feature_mask', type=str,
                        help='Use existing regressor to determine which parameters may be updated.  If no initial_regressor given, also used for initial weights')

    parser.add_argument('-i', '--initial_regressor', type=str, help='Initial regressor')
    parser.add_argument('--initial_weight', type=float, help='Set all weights to an initial value of 1')
    #parser.add_argument('--random_weights')
    #parser.add_argument('--input_feature_regularizer')

    #parser.add_argument('--span_server')
    #parser.add_argument('--unique_id')
    #parser.add_argument('--total')
    #parser.add_argument('--node')

    #parser.add_argument('-B', '--bootstrap')

    #Other options
    parser.add_argument('--audit_log_file', type=str, help='Audit log file')
    parser.add_argument('--summary_file', type=str, help='Summary file')
    parser.add_argument('--save_summary_to_file', action='store_true', help='Write summary to summary_file')

    args = parser.parse_args()
    arg_string = sys.argv[2:]

    #Remove Other options from args
    try:
        arg_string.remove('--save_summary_to_file')
        arg_string.remove('--audit_log_file')
        arg_string.remove(args.audit_log_file)
        arg_string.remove('--summary_file')
        arg_string.remove(args.summary_file)
    except ValueError:
        pass

    if args.testonly is None:
        train(args, arg_string)
    elif args.testonly:
        test(args, arg_string)
    else:
        sys.stderr.write('Invalid arguments: "%s" ' % arg_string)
        sys.exit(1)

def train(args, arg_string):
    """
    Learn the model
    :param args:
    :param arg_string:
    """
    vw = VW(args, arg_string)
    sys.stderr.write('Training with args: "%s" ' % ' '.join(arg_string))
    vw.learn()
    vw.summarize_features(audit_log=args.audit_log_file, summary_file=args.summary_file,
                          save_summary=args.save_summary_to_file)
    print 'Training_time\tAudit_time\tSparsity\n%s' % '\t'.join([str(round(vw.training_time)), str(round(vw.audit_time)), str(vw.sparsity)])

def test(args, arg_string):
    """
    Evaluate the model
    :param args:
    :param arg_string:
    """
    vw = VW(args, arg_string)
    sys.stderr.write('Testing with args: "%s" ' % ' '.join(arg_string))
    print "AUC\n%s" % vw.test


if __name__ == '__main__':
    main()