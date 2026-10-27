import argparse
import json
import os
from pathlib import Path
import time

from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

from .store import Store
from .streaming import MinuteBatch, Models


def create_topic(bootstrap, topic):
    admin = AdminClient({'bootstrap.servers': bootstrap})
    if topic not in admin.list_topics(timeout=10).topics:
        admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic].result()
    metadata = admin.list_topics(topic, timeout=10)
    if len(metadata.topics[topic].partitions) != 1:
        raise ValueError('Ordered replay requires exactly one Kafka partition')


def publish(bootstrap, topic, stream, batches):
    producer = Producer({'bootstrap.servers': bootstrap, 'enable.idempotence': True, 'acks': 'all',
                         'message.max.bytes': 8_000_000, 'delivery.timeout.ms': 30000})
    errors = []
    def delivered(error, message):
        if error:
            errors.append(str(error))
    for batch in batches:
        producer.produce(topic, partition=0, key=stream, value=batch.model_dump_json(), on_delivery=delivered)
        producer.poll(0)
    if producer.flush(35) or errors:
        raise RuntimeError('Kafka delivery failed: ' + str(errors))


def consume(bootstrap, topic, group, store, limit=None, crash_after_database=None, observer=None):
    create_topic(bootstrap, topic)
    consumer = Consumer({'bootstrap.servers': bootstrap, 'group.id': group,
                         'enable.auto.commit': False, 'enable.auto.offset.store': False,
                         'auto.offset.reset': 'earliest', 'max.poll.interval.ms': 300000})
    consumer.subscribe([topic])
    count = 0
    try:
        while limit is None or count < limit:
            message = consumer.poll(10)
            if message is None:
                raise TimeoutError('No next replay message within 10 seconds')
            if message.error():
                raise RuntimeError(str(message.error()))
            batch = MinuteBatch.model_validate_json(message.value())
            stream = message.key().decode('utf-8')
            started = time.perf_counter()
            result = store.apply(stream, batch)
            if observer is not None:
                observer({"sequence": batch.sequence, "duplicate": result["duplicate"],
                          "processing_seconds": time.perf_counter() - started,
                          "enqueue_to_database_seconds": time.time() - message.timestamp()[1] / 1000})
            count += 1
            if count == crash_after_database:
                raise RuntimeError('Injected failure after database commit, before Kafka acknowledgement')
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', default='127.0.0.1:19094')
    parser.add_argument('--topic', required=True)
    parser.add_argument('--group')
    parser.add_argument('--publish')
    parser.add_argument('--stream')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.publish:
        if not args.stream:
            parser.error('--publish requires --stream')
        create_topic(args.bootstrap, args.topic)
        batches = [MinuteBatch.model_validate_json(line) for line in Path(args.publish).read_text().splitlines()]
        publish(args.bootstrap, args.topic, args.stream, batches)
        return
    if not args.group:
        parser.error('Consumption requires --group')
    store = Store(os.environ['FRAUD_DSN'], Models(os.environ['FRAUD_MODELS']))
    consume(args.bootstrap, args.topic, args.group, store, args.limit)


if __name__ == '__main__':
    main()
