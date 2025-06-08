from PIL._tkinter_finder import tk
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def embed_plot_in_frame(self, fig, frame):
    """
    Osadza wykres matplotlib w ramce tkinter
    
    Args:
        fig: Obiekt Figure matplotlib
        frame: Ramka tkinter do osadzenia
    """
    try:
        print("DEBUG: PlotManager.embed_plot_in_frame wywoływane")

        if fig is None:
            print("ERROR: Figure jest None")
            return

        if frame is None:
            print("ERROR: Frame jest None")
            return

        print(f"DEBUG: Figure type: {type(fig)}")
        print(f"DEBUG: Frame type: {type(frame)}")

        # Wyczyść poprzednie wykresy
        for widget in frame.winfo_children():
            print(f"DEBUG: Usuwam widget: {widget}")
            widget.destroy()

        # Utwórz canvas z figurą
        print("DEBUG: Tworzę FigureCanvasTkAgg...")
        canvas = FigureCanvasTkAgg(fig, master=frame)

        print("DEBUG: Rysuję canvas...")
        canvas.draw()

        print("DEBUG: Umieszczam canvas w frame...")
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Dodaj toolbar nawigacji
        print("DEBUG: Dodaję toolbar...")
        try:
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            print("DEBUG: ✅ Toolbar dodany")
        except Exception as toolbar_error:
            print(f"WARNING: Nie można dodać toolbar: {toolbar_error}")
            # Toolbar nie jest krytyczny

        print("DEBUG: ✅ Wykres osadzony pomyślnie!")

    except Exception as e:
        print(f"ERROR: Błąd w embed_plot_in_frame: {str(e)}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")

        # Spróbuj prostsze podejście
        try:
            print("DEBUG: Próba prostszego podejścia...")
            for widget in frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            print("DEBUG: ✅ Prosty sposób zadziałał")

        except Exception as e2:
            print(f"ERROR: Nawet prosty sposób nie zadziałał: {e2}")
            raise e  # Re-raise original error