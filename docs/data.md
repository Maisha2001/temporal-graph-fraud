# Data provenance

The source is IBM's **HI-Small** transaction file, linked by
[IBM/AML-Data](https://github.com/IBM/AML-Data) to the
[IBM Transactions for Anti Money Laundering dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
IBM describes a simulated world with banks, people, businesses, legitimate
payments and laundering activity. It is synthetic, rather than anonymized
customer data. `Is Laundering` is a simulator-provided label, not a human
investigation outcome. This supports controlled graph experiments but does not
show effectiveness on real financial activity.

IBM explicitly distinguishes the repository's Apache license from the data's
[CDLA-Sharing-1.0 license](https://spdx.org/licenses/CDLA-Sharing-1.0.html).
Raw and modified data remain outside this code repository. If distributing such
tables separately, preserve IBM attribution and the data license, and identify
modifications. Public results here are aggregate computational outputs. The code
license does not relicense IBM's data.

Pinned file:

- `HI-Small_Trans.csv`, 475,664,283 bytes.
- SHA-256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.
- Downloaded directly from the publisher's Kaggle endpoint on October 25, 2026.
- 5,078,345 rows, with timestamps from September 1 through September 18, 2022.
- Source and destination are `(bank, account)` pairs. Bank numbers are
  canonicalized without leading zeroes; account identifiers are preserved as
  strings. Neither account nor bank identifiers are prediction features.

The CSV contains two `Account` column names; the parser checks the entire header
and assigns distinct source/target fields by position. File order is not temporal
order, so rows are stably ordered by timestamp and original row number. Duplicate
transaction-looking rows are retained: without a publisher transaction key,
identical records cannot safely be assumed to be duplicates.

Timestamps have minute resolution and no documented timezone in the file. They
are treated as relative simulator time, with no claim about a real local timezone.
The final days are sparse; the reserved evaluation must retain and separately
describe that tail rather than discard it after viewing performance.
