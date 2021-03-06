import sys
import os
import collections
import random
import struct
import tensorflow as tf
from tensorflow.core.example import example_pb2

processed_dir = "./processed"
chunks_dir = os.path.join(processed_dir, "chunked")

# We use these to separate the tweet sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
BRACKETS = {'-lrb-': '(',
            '-rrb-': ')',
            '-lcb-': '{',
            '-rcb-': '}',
            '-lsb-': '[',
            '-rsb-': ']',
            '``': '"',
            "''": '"'}

num_expected_articles = 33445

VOCAB_SIZE = 76409
CHUNK_SIZE = 100  # num examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = os.path.join(processed_dir, set_name + ".bin")
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' %
                                   (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack(
                    '%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def clean_summary(summary):
    word_list = summary.split(' ')
    idx = 0
    while idx < len(word_list):
        if word_list[idx] in BRACKETS:
            word_list[idx] = BRACKETS[word_list[idx]]
        idx += 1
    return ' '.join(word for word in word_list)


def clean_tweet(tweet):
    word_list = tweet.split(' ')
    # print(word_list)
    while word_list[0] == "rt":
        word_list = word_list[3:]
    if "rt" in word_list:
        i = word_list.index("rt")
        if word_list[i + 1][0] == '@':
            word_list = word_list[:i] + word_list[i+2]
    idx = 0
    while idx < len(word_list):
        if word_list[idx] == '' or word_list[idx] == "..." \
                or word_list[idx] == "pdf" or word_list[idx] == "doc" or word_list[idx] == "~":
            word_list = word_list[:idx] + word_list[idx + 1:]
            idx -= 1
        elif word_list[idx][0] == '#':
            idx += 1
            continue
        elif word_list[idx] == '[':
            jdx = idx
            while idx < len(word_list) and word_list[idx] != ']':
                idx += 1
            if idx == len(word_list):
                word_list = word_list[:jdx]
            else:
                word_list = word_list[:jdx] + word_list[idx+1:]
            idx = jdx - 1
        elif idx+1 < len(word_list) and word_list[idx] == '(' and word_list[idx+1] == "arxiv":
            jdx = idx - 1        # Remove the . as well before the arxiv link
            while idx < len(word_list) and word_list[idx][0] != ')':
                idx += 1
            idx += 1
            kdx = idx
            while idx < len(word_list) and word_list[idx][0] != '#':
                idx += 1
            if idx >= len(word_list):
                idx = kdx
            if jdx == -1:
                word_list = word_list[idx:]
                idx = jdx
            elif word_list[jdx] != '.':
                word_list = word_list[:jdx+1] + word_list[idx:]
                idx = jdx
            else:
                word_list = word_list[:jdx] + word_list[idx:]
                idx = jdx - 1
        idx += 1
    if word_list[-1] == '.':
        word_list = word_list[:-1]
    tweet = SENTENCE_START + ' '
    tweet += ' '.join(word for word in word_list)
    tweet += ' ' + SENTENCE_END
    return tweet


def write_to_bin(summaries, tweets, titles, line_nums, out_file, makevocab=False):
    """Reads the tokenized files and takes only the particular line numbers for usage writes them to a out_file."""
    print("Making bin file")
    num_articles = len(summaries)
    jdx = 0

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(summaries):
            if jdx == len(line_nums) or idx != line_nums[jdx]:
                continue
            if idx % 100 == 0:
                print("Writing %i of %i; %.2f percent done" %
                      (idx, num_articles, float(idx)*100.0/float(num_articles)))

            # Convert to lower case
            summary = clean_summary(summaries[idx].lower())
            tweet = clean_tweet(tweets[idx].lower())
            tweet = SENTENCE_START + ' ' + titles[idx].lower() + ' ' + SENTENCE_END + ' ' + tweet
            # print(tweet)
            if summary[-1] != '.':
                summary += ' .'
            # print(summary)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([
                                                                           summary.encode("utf-8")])
            tf_example.features.feature['abstract'].bytes_list.value.extend([
                tweet.encode("utf-8")])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                summary_tokens = summary.split(' ')
                tweet_tokens = tweet.split(' ')
                tweet_tokens = [t for t in tweet_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = summary_tokens + tweet_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)
            jdx += 1

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(processed_dir, "vocab"), 'w', encoding="utf-8") as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def dataset_split(line_nums):
    random.seed(42)
    random.shuffle(line_nums)
    split_1 = int(0.9 * num_expected_articles)
    split_2 = int(0.95 * num_expected_articles)
    # print(split_1, split_2)
    train_line_nums = sorted(line_nums[:split_1])
    val_line_nums = sorted(line_nums[split_1:split_2])
    test_line_nums = sorted(line_nums[split_2:])
    print(len(train_line_nums), len(val_line_nums), len(test_line_nums))
    return train_line_nums, val_line_nums, test_line_nums


def check_dataset_dir(dataset_dir):
    tokenized = os.listdir(dataset_dir)
    if not ("summaries_tokenized" in tokenized and "titles_tokenized" in tokenized and "tweets_tokenized" in tokenized):
        raise Exception(
            "Dataset directory does not contain all the tokenized files")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python make_datafiles.py <path_to_dataset_containing_<summaries_tokenized>_<titles_tokenized>_and_<tweets_tokenized>>")
        sys.exit()
    dataset_dir = sys.argv[1]

    # Check if the dataset directory contains <summaries_tokenized>, <titles_tokenized> and <tweets_tokenized>
    check_dataset_dir(dataset_dir)

    # Create chunks directory
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    line_nums = [i for i in range(num_expected_articles)]
    print(len(line_nums))
    train_line_nums, val_line_nums, test_line_nums = dataset_split(line_nums)
    summaries = read_text_file(os.path.join(
        dataset_dir, "summaries_tokenized"))
    tweets = read_text_file(os.path.join(dataset_dir, "tweets_tokenized"))
    titles = read_text_file(os.path.join(dataset_dir, "titles_tokenized"))
    write_to_bin(summaries, tweets, titles, test_line_nums,
                 os.path.join(processed_dir, "test.bin"))
    write_to_bin(summaries, tweets, titles, val_line_nums,
                 os.path.join(processed_dir, "val.bin"))
    write_to_bin(summaries, tweets, titles, train_line_nums,
                 os.path.join(processed_dir, "train.bin"), makevocab=True)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 100 examples, and saves them in finished_files/chunks
    chunk_all()
