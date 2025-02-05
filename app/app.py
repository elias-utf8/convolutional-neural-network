import customtkinter as ctk
from tkinter import filedialog
from customtkinter import CTkImage
from PIL import Image

class CNN_App:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.title("Convolutiv Neural Network")
        self.root.geometry("600x500")
        #self.root.resizable(False, False)
        self.dataset_var = ctk.StringVar(value="Select Model")

        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(side="left", padx=10, pady=10, fill="y")

        """------ Dropdown pour choisir le modèle ------"""
        self.dataset_dropdown = ctk.CTkOptionMenu(
            master=control_frame,
            values=["CIFAR10", "CIFAR100", "COCO", "MobilNet (recommanded)"],
            variable=self.dataset_var,
            width=200,
            fg_color="grey",
            button_color="grey",
            text_color="white"
        )
        self.dataset_dropdown.pack(pady=10, anchor="w")

        """------ Bouton pour confirmer la sélection ------"""
        select_button = ctk.CTkButton(
            master=control_frame, 
            text="Confirm Selection", 
            command=self.ChoisirModele,
            width=100,
            fg_color="grey"
        )
        select_button.pack(pady=0, anchor="w")

        """------ Séparateur ------"""
        separator = ctk.CTkFrame(main_frame, width=2, fg_color="white")
        separator.pack(side="left", fill="y", padx=10)

        """------ Zone de texte ------"""
        self.text_area = ctk.CTkTextbox(
            master=main_frame, 
            width=300,
            height=500,
            state="disabled"
        )
        self.text_area.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        """------ Bouton pour sélectionner une image ------"""
        select_image = ctk.CTkButton(
            master=control_frame, 
            text="Choisir image", 
            command=self.ChargerImage,
            width=120
        )
        select_image.pack(pady=40, anchor="center", expand=False)

        # Label pour afficher l'image
        self.label_image = ctk.CTkLabel(
            master=main_frame,  # Assurez-vous que main_frame est le bon parent
            text="Image va apparaître ici",  # Texte par défaut
            width=200,  # Largeur fixe pour le label
            height=200  # Hauteur fixe pour le label
        )
        self.label_image.pack(padx=30, pady=50, side="left")

    def Charger_CIFAR10(self):
        pass

    def Charger_CIFAR100(self):
        pass

    def Charger_COCO(self):
        pass

    def ChargerMobilNET(self):
        pass

    def ChoisirModele(self):
        selected_dataset = self.dataset_var.get()
        if selected_dataset == "Select Model":
            return

        if "(recommanded)" in selected_dataset:
            selected_dataset = selected_dataset[:-14]

        info_text = f"Selected dataset: {selected_dataset}\n\nInformation about {selected_dataset}..."
        self.ChangerTexte(info_text)

        if selected_dataset == "CIFAR10":
            self.Charger_CIFAR10()
        elif selected_dataset == "CIFAR100":
            self.Charger_CIFAR100()
        elif selected_dataset == "COCO":
            self.Charger_COCO()
        elif selected_dataset == "MobilNet":
            self.ChargerMobilNET()

    def ChangerTexte(self, text):
        self.text_area.configure(state="normal")
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", text) 
        self.text_area.configure(state="disabled")

    def ChargerImage(self):
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if filepath:
            img = Image.open(filepath)
            img_resized = img.resize((200, 200), Image.Resampling.LANCZOS)

            # Utilisation correcte de CTkImage avec une référence persistante
            self.image_ctk = CTkImage(light_image=img_resized, size=(200, 200))
            
            self.label_image.configure(image=self.image_ctk, text="")  
            self.ChangerTexte("Image chargée avec succès.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CNN_App()
    app.run()
