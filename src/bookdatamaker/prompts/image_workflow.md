# Image Workflow
When you see image references like ![](images/0.jpg) in page content:
1. You SHOULD call list_page_images to get absolute file paths
2. You SHOULD call minimax_understand_image with the file path as image_url
3. You SHOULD incorporate the image analysis into your conversation
