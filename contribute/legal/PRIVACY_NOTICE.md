---
version: 2026-11-18-draft-2
status: draft
note: Draft, not in force. Written by a non-lawyer for Logan and a lawyer to review.
---
# Litmus contribute site: Privacy Notice

Version 2026-11-18-draft-2. This notice is part of the [Terms of Use](TERMS_OF_USE.md).

**Who is responsible.** Logan Hallee, an individual ("we", "us"), operates the Litmus contribute site (the "Service") and decides what is stored and why. Contact: open an issue at https://github.com/lhallee/litmus/issues. Issues are public, so anyone can read them: please do not put private details in one, or a text you want removed (use the removal form for that).

**In short.** We store the texts people check, with the model's reading and any label, privately, to improve Litmus. We never publish them or release them in any dataset. Please do not paste sensitive or personal information, yours or anyone else's: you are responsible for what you paste. We do not store who sent a text: no address, account, cookie or device detail, though a text can itself point to who wrote it or who it is about. That protects you, and it also means we cannot find "your" texts for you; we find a text by its content instead, and anyone can have a text deleted by sending the same text through the removal form.

## What we store

For each text checked on the Service:

- the text, exactly as sent;
- its fingerprint (a SHA-256 hash of the text);
- a random reference for it, which also encodes the time it was received;
- the model's reading of it and the model's version;
- the label you give, if any (AI, mixed or human), and when you gave or changed it;
- when it was received;
- the version of the Terms of Use and this notice that you agreed to by ticking the box beside the Check button (the Service accepts no text without that box ticked);
- how many times the same text was sent under the same model and terms version, and how many of those sends carried each label.

For a removal request sent through the form: the fingerprint of the text to remove and the time, never the text itself.

If you open an issue to contact us, it is public on GitHub under your GitHub account, and GitHub keeps it under its own terms. We keep it as a record of what we did.

To let you add or change a label, the page is given the stored text's random reference, and the Service accepts a label for that text for 30 minutes only. The reference points to the text, not to you, and it is not a cookie.

## What we do not store

- Your IP address, your browser or device, or your location. We keep no record of who sent a text. The text itself is stored, and its content or style may still point to its author or to the people it mentions.
- Cookies or any other identifier of you in your browser. The Service sets none.
- An account. There is none.

To stop abuse, the server counts requests per address for a short time. It does this with a salted one-way hash of the address, kept only in the memory of the running server; the salt is random, never written down, and lost when that server stops, normally within minutes of the last request. The hash is never written to storage or to logs.

## Please do not paste sensitive or personal information

Please do not paste sensitive or personal information, yours or anyone else's (Terms of Use, sections 4.3 and 6.2). You are responsible for what you paste. If a text does contain such information, we hold it as part of the text even though we do not know who sent it, and we never publish it. We may remove personal information we find, but we do not promise to look for it. Anyone can have a text removed (below).

## Why we use it

- To show you the reading you asked for.
- To improve Litmus: to train, calibrate and evaluate its models. We may publish those models, and figures computed from many texts that contain no text, such as counts and error rates. We never publish the texts, or the readings and labels stored with them.
- For research on detecting model-written text, with the texts kept private.
- To keep the Service working and to stop abuse.

We use a text because you agreed to this use by ticking the box beside the Check button. The Service is offered on the same terms to everyone, wherever they are, and anyone may withdraw that agreement at any time by having the text deleted with the removal form (below). Withdrawing does not affect what we did with the text before it was deleted.

## Never published

We never publish the texts sent to the Service, in whole or in part, or release them in any dataset, and we never publish the readings or labels stored with them. They stay in the private storage described below.

## Who else handles it

- **Modal Labs, Inc.** runs the servers and the storage the Service uses. To deliver your request, Modal receives it, including your IP address and the text you send, and it may keep its own operational logs under its own terms. We do not control those logs and do not use them to identify anyone.
- **Hugging Face, Inc.** hosts our dataset of stored texts, privately; we never make it public. It keeps earlier versions of the dataset until we rewrite the dataset's history, which we do whenever we delete a removed text from it (see "Removal" below).
- **GitHub, Inc.** hosts the private page and the issue tracker we use for contact. On the private page your text is scored in your browser and is not sent to us or to anyone. GitHub receives and logs the IP address of each visitor to the page, and keeps issues, which are public, under its own privacy statement.

We do not sell or rent any data, show advertising, or share data for advertising.

## How long we keep it

Texts sit in working storage on Modal until we compile them into our private dataset on Hugging Face, which we do at least once every four weeks, and are then deleted from working storage. We keep the private dataset, and the copies we make on our own computers to train and evaluate models, indefinitely, until we delete them or a text is removed. Removal fingerprints are kept in the private dataset for as long as it exists, so that removed texts stay out of it. A fingerprint cannot be read back into its text, but someone who already has, or can guess, the exact text could use it to confirm that the text was removed; the fingerprints are private, like the dataset.

## Removal, and your other rights

Because nothing links a text to the person who sent it, we cannot tell which texts are yours, so we cannot show you, correct or send you "your" texts. What we can do is find a text by its content: send the exact text through the removal form on the Service, and we delete it from working storage, from our private dataset including its earlier versions, and from our own copies, within 30 days of the request, and keep it out of the dataset in future, even if it is sent again. Anyone can do this, for any text, at any time; it is how you withdraw your agreement. Removal does not change a model that was already trained, or records our hosting providers keep under their own terms. If the stored version differs from yours, or if someone else's text mentions you and you do not have it, open an issue describing it in general terms, without the private details, and we will search for it.

You may also ask us about our use of texts, or complain to a data protection authority.

## Children

The Service is for people 18 and older and is not directed to children. If you are under 18, use the private page, which sends us nothing. If you believe a child has sent a text, send the same text through the removal form and we will delete it as described above.

## Security

Everything travels to and from the Service encrypted (HTTPS). Stored texts are kept in private storage that only we, and our hosting providers as they run it, can reach, and our dataset is private and never made public.

## Where it is processed

We run the Service from the United States. Your text is sent to our hosting providers, which may process and store it in the United States or in other countries where they run their systems.

## Changes

We may change this notice by posting a new version with a new version label. Each stored text records the version in force when it was sent.

---

Parts of this notice are adapted from the 37signals (Basecamp) Privacy Policy, https://github.com/basecamp/policies, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); we changed and shortened them. Other parts are adapted from GitHub's site policies (CC0 1.0). Neither 37signals nor GitHub endorses this notice.
