import customtkinter as ctk

class CNN_App:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.title("Convolutiv Neural Network")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.dataset_var = ctk.StringVar(value="Select Model")

        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(side="left", padx=10, pady=10, fill="y")

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

        select_button = ctk.CTkButton(
            master=control_frame, 
            text="Confirm Selection", 
            command=self.ChoisirModele,
            width=120
        )
        select_button.pack(pady=10, anchor="w")

        separator = ctk.CTkFrame(main_frame, width=2, fg_color="white")
        separator.pack(side="left", fill="y", padx=10)

        self.text_area = ctk.CTkTextbox(
            master=main_frame, 
            width=300,
            height=500,
            state="disabled"
        )
        self.text_area.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    def Charger_CIFAR10():
        return None
    def Charger_CIFAR100():
        return None
    def Charger_COCO():
        return None
    def CHargetMobilNET():
        return None

    def ChoisirModele(self):
        selected_dataset = self.dataset_var.get()
        if selected_dataset == "Select Model":
            return None;
        if ("(recommanded)" in selected_dataset):
            selected_dataset=selected_dataset[:-14]
        info_text = f"Selected dataset: {selected_dataset}\n\nInformation about {selected_dataset}..."
        self.ChangerTexte(info_text)

        if selected_dataset=="CIFAR10":
            CharerCIFAR10()
        elif selected_dataset=="CIFAR100":
            ChargerCIFAR100()
        elif selected_dataset=="COCO":
            ChargerCOCO()
        elif selected_dataset=="MobilNet":
            ChargerMobilNET()

    def ChangerTexte(self, text):
        self.text_area.configure(state="normal")
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", text) 
        self.text_area.configure(state="disabled")



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CNN_App()
    app.run()
