from transformers import BertTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_string(string, max_length=12):
    """Tokenizes the string."""

    string = string.replace(".", " ")

    original_tokens = tokenizer.tokenize(string)
    if len(original_tokens) > max_length:
      print(f"Tokens trimmed for: {string}")
      trimmed_tokens = original_tokens[max_length:]
      print(f"Trimmed tokens: {trimmed_tokens}")

    tokens = tokenizer(string,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=False, # Optional: If you don't need attention_mask, set to False for slight efficiency
                    return_token_type_ids=False # Optional: If you don't need token_type_ids, set to False
                    )

    return tokens['input_ids']

# def identify_trimmed_tokens(string, max_length=12):
#     """Identifies tokens that are trimmed during BERT tokenization."""

#     string = string.replace(".", " ")

#     # Tokenize the string without truncation to get all tokens
#     original_tokens = tokenizer.tokenize(string)
#     print(f"Original tokens: {original_tokens}")

#     # Tokenize the string with truncation
#     tokens = tokenizer(string,
#                     padding='max_length',
#                     truncation=True,
#                     max_length=max_length,
#                     return_attention_mask=False,
#                     return_token_type_ids=False)
#     print(f"tokens['input_ids']: {tokens['input_ids']}")

#     input_ids = tokens['input_ids']
#     truncated_tokens_ids = input_ids[1:-1] # remove [CLS] and [SEP]
#     truncated_tokens = tokenizer.convert_ids_to_tokens(truncated_tokens_ids)
#     print(f"Truncated tokens: {truncated_tokens}")


#     if len(original_tokens) > max_length: # account for [CLS] and [SEP]
#         trimmed_tokens = original_tokens[max_length:]
#         print(f"Tokens were trimmed for: {string}")
#         print(f"Trimmed tokens: {trimmed_tokens}")
#         return trimmed_tokens
#     else:
#         print("No tokens were trimmed.")
#         return []

# # Example usage
# string_to_tokenize = "This is a long string to test the tokenization and identify trimmed tokens."
# max_length_value = 12
# trimmed_tokens = identify_trimmed_tokens(string_to_tokenize, max_length_value)
# print(f"Tokens trimmed from the input string: {trimmed_tokens}")

# string_to_tokenize = "Short string"
# max_length_value = 12
# trimmed_tokens = identify_trimmed_tokens(string_to_tokenize, max_length_value)
# print(f"Tokens trimmed from the input string: {trimmed_tokens}")

# # Example DOM node
# node = {
#     'class_name': 'android.widget.Button',
#     'resource_id': 'com.example.app:id/submit_button',
#     'text': 'Submit'
# }

# # Tokenize the node
# # input_ids = tokenize_dom_node(node['class_name'], node['resource_id'], node['text'])
# # input_ids = tokenize_dom_node_64(node)
# input_ids = tokenize('android.widget.LinearLayout')
# print(input_ids, len(input_ids))

# input_ids = tokenize_string('androidx.viewpager.widget.ViewPager', max_length=25)
# print(input_ids, len(input_ids))

# input_ids = tokenize('android.widget.ScrollView')
# print(input_ids, len(input_ids))

# input_ids = tokenize('android.view.View')
# print(input_ids, len(input_ids))

# input_ids = tokenize('android.view.ViewGroup')
# print(input_ids, len(input_ids))

# input_ids = tokenize('android.widget.TextView')
# print(input_ids, len(input_ids))


# '''class: hierarchy
# text None
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.LinearLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.LinearLayout
# text
# resource_id None
# class: android.widget.LinearLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.LinearLayout
# text
# resource_id None
# class: android.widget.LinearLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id None
# class: android.widget.LinearLayout
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id android:id/content
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/launcher
# class: android.widget.FrameLayout
# text
# resource_id android:id/content
# class: android.widget.FrameLayout
# text
# resource_id android:id/content
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/launcher
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.view.View
# text
# resource_id com.google.android.apps.nexuslauncher:id/scrim_view
# class: android.widget.ScrollView
# text
# resource_id com.google.android.apps.nexuslauncher:id/workspace
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.view.View
# text
# resource_id com.google.android.apps.nexuslauncher:id/scrim_view
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/drag_layer
# class: android.view.View
# text
# resource_id com.google.android.apps.nexuslauncher:id/scrim_view
# class: android.view.View
# text
# resource_id com.google.android.apps.nexuslauncher:id/scrim_view
# class: android.widget.ScrollView
# text
# resource_id com.google.android.apps.nexuslauncher:id/workspace
# class: android.view.View
# text
# resource_id None
# class: android.widget.ScrollView
# text
# resource_id com.google.android.apps.nexuslauncher:id/workspace
# class: android.view.ViewGroup
# text
# resource_id None
# class: android.view.ViewGroup
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/search_container_workspace
# class: android.widget.TextView
# text Gmail
# resource_id None
# class: android.widget.TextView
# text Gmail
# resource_id None
# class: android.view.ViewGroup
# text
# resource_id None
# class: android.view.ViewGroup
# text
# resource_id None
# class: android.view.ViewGroup
# text
# resource_id None
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/search_container_workspace
# class: android.widget.FrameLayout
# text
# resource_id com.google.android.apps.nexuslauncher:id/search_container_workspace'''