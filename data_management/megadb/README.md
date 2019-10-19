

# MegaDB

Internally we store all labels and metadata associated with each image sequence in a NoSQL database for easy querying.

For images whose sequence information is unknown, they will be contained in separate sequences whose `seq_id` will start with `dummy_`.

