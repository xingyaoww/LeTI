from typing import Set, List, Tuple, Dict, Any

def convert_event_obj_to_dict(
    event_var,
    split_role_value=True,
    raise_error=False
):
    event_var = event_var.__dict__
    event_dict = {}
    for role, objs in event_var.items():
        cur_role_values = []

        if not hasattr(objs, "__iter__"):
            if raise_error:
                raise ValueError(
                    f"Expecting a list of Entity instance for role {role}, but got {type(objs)}"
                )
            continue

        for obj in objs:
            if hasattr(obj, "__dict__") and "name" in obj.__dict__:
                val = obj.__dict__["name"]
                if split_role_value:
                    val = val.split()
                cur_role_values.append(
                    (obj.__class__.__name__, val)
                )
            elif isinstance(obj, str):
                if split_role_value:
                    obj = obj.split()
                cur_role_values.append((obj.__class__.__name__, obj))
                if raise_error:
                    raise ValueError(
                        f"Expecting a list of Entity instance for role {role} with .name attribute, but got a list of {type(obj)}"
                    )
            elif raise_error:
                raise ValueError(
                    f"Expecting a list of Entity instance for role {role} with .name attribute, but got a list of {type(obj)}"
                )
        role = role[0].upper() + role[1:] # capitalize the first letter
        event_dict[role] = cur_role_values
    return event_dict

def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i

    arg_head = cur_i

    return (arg_head, arg_head)

def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False, doc=None):
    match = None
    arg_len = len(arg)
    min_dis = len(context_words)  # minimum distance to trigger
    for i, w in enumerate(context_words):
        if context_words[i:i+arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start-i-arg_len)
            else:
                dis = abs(i-trigger_end)
            if dis < min_dis:
                match = (i, i+arg_len-1)
                min_dis = dis

    if match and head_only:
        assert (doc != None)
        match = find_head(match[0], match[1], doc)
    return match

def construct_pred_set(
    predicted_args,
    cur_event,
    context_words,
    doc,
    head_only=True,
    nlp=None
):
    # get trigger
    # extract argument span
    trigger_start = cur_event['trigger']['start']
    trigger_end = cur_event['trigger']['end']
    predicted_set = set()

    lowercased_context_words = [w.lower() for w in context_words]
    lowercased_doc = nlp(' '.join(lowercased_context_words)
                         ) if head_only else None

    not_matched_pred_args = []
    for argname in predicted_args:
        # this argument span is inclusive, FIXME: this might be problematic
        for entity_type, entity_text in predicted_args[argname]:
            if entity_text is None:  # means no entity is found for this argument
                continue
            entity_text: List[str]
            arg_span = find_arg_span(
                entity_text,
                context_words,
                trigger_start,
                trigger_end,
                head_only=head_only,
                doc=doc
            )

            # Attempt to fixed due to cases or symbols in the text
            # e.g., entity = "anwar" vs context_words = "Anwar"
            # e.g., entity = ["Anne-Marie"] vs context_words = ["Anne", "-", "Marie"]
            # e.g., entity = ["roh", "moo-hyun"] vs context_words = ["roh", "moo", "-", "hyun"]
            if not arg_span:
                normalized_entity_text = []
                for word in entity_text:
                    word = word.lower()
                    # process hyphenated words
                    if "-" in word and len(word) > 1:
                        normalized_entity_text.extend(
                            word.replace("-", " - ").split())
                    else:
                        normalized_entity_text.append(word)
                    # TODO: If we really want higher performance on ACE05,
                    # we could fix U.S. -> U.S, british -> british, etc.
                arg_span = find_arg_span(
                    normalized_entity_text,
                    lowercased_context_words,
                    trigger_start,
                    trigger_end,
                    head_only=head_only,
                    doc=lowercased_doc
                )

            if arg_span:  # if None means hullucination
                predicted_set.add(
                    (arg_span[0], arg_span[1],
                     cur_event["event_type"], argname, entity_type)
                )
            else:
                not_matched_pred_args.append({
                    "role": argname,
                    "entity_type": entity_type,
                    "text": entity_text
                })
            # With code generation, we don't need to care for "and"
    return predicted_set, not_matched_pred_args

def get_entity(entity_mentions, entity_id):
    for ent in entity_mentions:
        if ent['id'] == entity_id:
            return ent

def clean_span(tokens, span):
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0] != span[1]:
            return (span[0]+1, span[1])
    return span

def construct_gold_set(tokens, doc, cur_event, entity_mentions, head_only=True):
    gold_set = set()
    # set of canonical mention ids, singleton mentions will not be here
    gold_canonical_set = set()
    for arg in cur_event['arguments']:
        argname = arg['role']
        entity_id = arg['entity_id']
        entity = get_entity(entity_mentions, entity_id)

        span = (entity["start"], entity["end"]-1)  # convert to inclusive span
        # clean up span by removing `a` `the`
        span = clean_span(tokens, span)

        if head_only and span[0] != span[1]:
            span = find_head(span[0], span[1], doc=doc)

        gold_set.add(
            (span[0], span[1], cur_event["event_type"], argname, entity["entity_type"]))
    return gold_set, gold_canonical_set

def match_pred_gold_sets(
    predicted_set: Set[Tuple[int, int, str, str]],
    gold_set: Set[Tuple[int, int, str, str]],
    tokens: List[str] = None, # only needed for raising error
    raise_error_when_not_matched: bool = False, # NOTE: this is only for execution debugging
):
    stats = {
        "arg_idn_num": 0,
        "arg_class_num": 0,
        "arg_ner_num": 0,
        "arg_class+ner_num": 0,
    }

    correctly_identified = set()
    correctly_classified = set()
    for pred_arg in predicted_set:
        arg_start, arg_end, event_type, role, entity_type = pred_arg
        if tokens is not None:
            cur_role_value = " ".join(tokens[arg_start:arg_end+1])

        # 1. Check Argument Identification (Span + Event Type)
        gold_idn = {
            item for item in gold_set
            if item[0] == arg_start and item[1] == arg_end  # span matched
            and item[2] == event_type  # event type matched
        }

        if gold_idn:
            # Identification is correct
            stats["arg_idn_num"] += 1
            if raise_error_when_not_matched:
                correctly_identified.add(cur_role_value)
            # cur_arg_stat["correct_identification"] = True

            # 2. Check Argument Classification (Span + Event Type + Role)
            gold_class = {
                item
                for item in gold_idn
                if item[3] == role  # role matched
            }
            if gold_class:
                # an gold argument is indentified and assigned to correct role
                stats["arg_class_num"] += 1
                if raise_error_when_not_matched:
                    correctly_classified.add((role, cur_role_value))

            elif raise_error_when_not_matched:
                raise ValueError(
                    f"A successfully identified argument \"{cur_role_value}\" is assigned to wrong role \"{role}\"")
            
            # 3. Check Argument Identification + NER (Span + Event Type + Entity Type)
            gold_ner = {
                item
                for item in gold_idn
                if item[4] == entity_type  # entity type matched
            }
            if gold_ner:
                # an gold argument is indentified and assigned to correct role
                stats["arg_ner_num"] += 1

            # 4. Check Argument Classification + NER (Span + Event Type + Role + Entity Type)
            gold_cls_ner = {
                item
                for item in gold_class
                if item[4] == entity_type  # entity type matched
            }
            if gold_cls_ner:
                # an gold argument is indentified and assigned to correct role
                stats["arg_class+ner_num"] += 1
        elif raise_error_when_not_matched:
            raise ValueError(
                f"An identified argument \"{cur_role_value}\" does not match any ground truth argument."
            )

    gold_arg_num = len(gold_set)
    if raise_error_when_not_matched:
        if stats["arg_idn_num"] != gold_arg_num:
            raise ValueError(
                f"Only {stats['arg_idn_num']} out of {gold_arg_num} arguments are correctly identified: {correctly_identified}"
            )
    
        if stats["arg_class_num"] != gold_arg_num:
            # I think this will never be triggered?
            raise ValueError(
                f"Only {stats['arg_class_num']} (correctly identified) out of {gold_arg_num} arguments are correctly classified: {'='.join(pair) for pair in correctly_classified}"
            )
    
    return stats
