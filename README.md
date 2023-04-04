# J_Captioneer_v2.release



# J_Captioneer

J_Captioneer_v1 is a PyQt5-based image editor and browser, as well as a caption generator and editor that allows users to view, navigate, and manage images and captions with ease. This streamlines your workflow and makes it easy to edit the captions to your training images. With J_Captioneer, you can quickly and efficiently browse through a directory of images. crop and resize them, create and edit the captions to each one. The app automatically loads all the images and their corresponding text files, and provides a simple and intuitive layout for editing captions. The app also has two find/replace functions as well as a prefix/suffix function.

# What's New in v2?

1. BLIP & VIT-GPT2: Generate captions using the state-of-the-art models BLIP and VIT-GPT2.
2. Crop & Resize: Crop and resize images to your desired dimensions, similar to the functionality found on birme.net.
3. Settings: Launch the app in dark mode, remember the last directory opened, or choose a default directory.
4. Removed Find X and add after/before: This function can now be achieved with the find and replace function.
5. Optimized the program: Faster load times and no more frozen windows!

## Features

J_Captioneer_v2 builds on the success of J_Captioneer_v1, with the following features:

1. Load and display images in a directory.
2. Navigate through images with left and right arrow buttons or keys.
3. Add, edit, generate, and save captions as separate text files associated with each image.
4. Display images as thumbnails for easy navigation.
5. Add prefix and suffix to all captions in a directory.
6. Find and replace text in all captions in a directory.
7. Crop and resize images with a user-friendly interface.
8. Toggle dark mode for a more comfortable viewing experience.
9. Configure settings such as launching dark mode on startup and remembering the last directory.

## Installation

### Download and Extract
Download the repository as a ZIP file from the releases page or clone it using the command git clone https://github.com/sjackp/J_Captioneer.v2.release.git.
Extract the ZIP file to any folder of your choice.
Run the .exe

### From Source
Clone the repository via git clone https://github.com/sjackp/J_Captioneer.v2.release.git
Run the .exe

## Usage

To use J_Captioneer, follow these steps:

1. Launch the application by running `J_Captioneer.exe` (standalone executable)
2. Click "Choose Directory" to select the folder containing the images you want to work with.
3. Thumbnails of the images will be displayed. Click on a thumbnail to view the image and its associated caption (an empty .txt file will be created if unavailable).
4. Click on the dropdown menu next to "Model:" to select the img-to-txt model (VIT-GPT2 or BLIP) and Click "Generate Captions For All" to generate captions.
5. Click on "Generate Caption" button to generate a caption only for the image you are viewing.  
6. Use the left and right arrow buttons or keys to navigate between images.
7. Edit the caption in the text box and click "Save" to save the changes(or Ctrl+s). Captions are saved in the text files with the same name as the image.
8. To return to the thumbnail view, click the "Back" button or press the "Escape" key.
9. Use the menu options under "File" to access additional features such as adding prefix/suffix, find and replace, toggling dark mode, and settings.

## Contributing

If you'd like to contribute to J_Captioneer, please [fork the repository](https://github.com/sjackp/J_Captioneer.v2.release/fork), create a new branch for your changes, and open a pull request.


## Credits

Seif Jackson+GPT-4
Daniel Knowlton
