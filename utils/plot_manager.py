"""
Zarządzanie wykresami matplotlib w interfejsie tkinter
Wersja 1.3 - z lepszym debugiem i fallback methods
"""

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

class PlotManager:
    """Zarządza osadzaniem wykresów matplotlib w interfejsie tkinter"""

    def __init__(self):
        self.current_canvas = None
        self.current_toolbar = None

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
            print(f"DEBUG: Figure size: {fig.get_size_inches()}")

            # Wyczyść poprzednie wykresy
            for widget in frame.winfo_children():
                print(f"DEBUG: Usuwam widget: {widget}")
                widget.destroy()

            # Reset poprzednich referencji
            self.current_canvas = None
            self.current_toolbar = None

            print("DEBUG: Tworzę FigureCanvasTkAgg...")

            # Utwórz canvas z figurą
            canvas = FigureCanvasTkAgg(fig, master=frame)
            print("DEBUG: Canvas utworzony")

            print("DEBUG: Rysuję canvas...")
            canvas.draw()
            print("DEBUG: Canvas narysowany")

            print("DEBUG: Umieszczam canvas w frame...")
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            print("DEBUG: Canvas umieszczony")

            # Zapisz referencję
            self.current_canvas = canvas

            # Dodaj toolbar nawigacji
            print("DEBUG: Dodaję toolbar...")
            try:
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                self.current_toolbar = toolbar
                print("DEBUG: ✅ Toolbar dodany")
            except Exception as toolbar_error:
                print(f"WARNING: Nie można dodać toolbar: {toolbar_error}")
                # Toolbar nie jest krytyczny, kontynuuj

            print("DEBUG: ✅ Wykres osadzony pomyślnie!")

        except Exception as e:
            print(f"ERROR: Błąd w embed_plot_in_frame: {str(e)}")
            import traceback
            print(f"TRACEBACK: {traceback.format_exc()}")

            # Spróbuj prostsze podejście bez toolbar
            try:
                print("DEBUG: Próba prostszego podejścia bez toolbar...")

                # Wyczyść ramkę
                for widget in frame.winfo_children():
                    widget.destroy()

                # Prosty canvas bez toolbar
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                self.current_canvas = canvas
                print("DEBUG: ✅ Prosty sposób zadziałał")

            except Exception as e2:
                print(f"ERROR: Nawet prosty sposób nie zadziałał: {e2}")

                # Ostatnia próba - pokaż w osobnym oknie
                try:
                    print("DEBUG: Próba wyświetlenia w osobnym oknie...")
                    # Konwertuj Figure na plt figure jeśli potrzeba
                    if hasattr(fig, 'show'):
                        fig.show()
                        print("DEBUG: ✅ Wykres wyświetlony w osobnym oknie")
                    else:
                        plt.figure(fig.number)
                        plt.show()
                        print("DEBUG: ✅ Wykres wyświetlony przez plt.show()")
                except Exception as e3:
                    print(f"ERROR: Nie można wyświetlić wykresu w żaden sposób: {e3}")

                raise e  # Re-raise original error

    def create_test_plot(self, frame):
        """Tworzy prosty wykres testowy"""
        try:
            print("DEBUG: Tworzę wykres testowy...")

            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            # Prosty wykres testowy
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, 'b-', label='sin(x)')
            ax.set_title('Wykres Testowy')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
            ax.legend()

            fig.tight_layout()

            # Osadź w frame
            self.embed_plot_in_frame(fig, frame)
            print("DEBUG: ✅ Wykres testowy utworzony")

        except Exception as e:
            print(f"ERROR: Błąd tworzenia wykresu testowego: {e}")

    def clear_plot(self, frame):
        """Czyści wykres z ramki"""
        try:
            print("DEBUG: Czyszczę wykres...")

            for widget in frame.winfo_children():
                widget.destroy()

            self.current_canvas = None
            self.current_toolbar = None

            # Dodaj placeholder
            placeholder = tk.Label(frame, text="📊 Miejsce na wykres\n\nWybierz analizę aby wyświetlić wykres",
                                   justify=tk.CENTER, font=('Arial', 12))
            placeholder.pack(expand=True)

            print("DEBUG: ✅ Wykres wyczyszczony")

        except Exception as e:
            print(f"ERROR: Błąd czyszczenia wykresu: {e}")

    def save_current_plot(self, filename=None):
        """Zapisuje aktualny wykres do pliku"""
        try:
            if self.current_canvas is None:
                print("WARNING: Brak aktualnego wykresu do zapisania")
                return False

            if filename is None:
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    title="Zapisz wykres",
                    defaultextension=".png",
                    filetypes=[
                        ("PNG files", "*.png"),
                        ("PDF files", "*.pdf"),
                        ("SVG files", "*.svg"),
                        ("All files", "*.*")
                    ]
                )

            if filename:
                self.current_canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"DEBUG: ✅ Wykres zapisany do: {filename}")
                return True

            return False

        except Exception as e:
            print(f"ERROR: Błąd zapisywania wykresu: {e}")
            return False

    def get_plot_info(self):
        """Zwraca informacje o aktualnym wykresie"""
        if self.current_canvas is None:
            return "Brak aktualnego wykresu"

        fig = self.current_canvas.figure
        info = f"""
📊 INFORMACJE O WYKRESIE:
• Typ: {type(fig).__name__}
• Rozmiar: {fig.get_size_inches()} cali
• DPI: {fig.dpi}
• Liczba subplot: {len(fig.axes)}
• Tytuł: {fig._suptitle.get_text() if fig._suptitle else 'Brak'}
"""
        return info

    def configure_matplotlib(self):
        """Konfiguruje matplotlib dla lepszej kompatybilności z tkinter"""
        try:
            # Użyj backend kompatybilnego z tkinter
            plt.switch_backend('TkAgg')
            print("DEBUG: ✅ Backend matplotlib ustawiony na TkAgg")

            # Ustaw style
            try:
                plt.style.use('seaborn-v0_8')
                print("DEBUG: ✅ Styl seaborn-v0_8 zastosowany")
            except:
                try:
                    plt.style.use('default')
                    print("DEBUG: ✅ Styl default zastosowany")
                except:
                    print("WARNING: Nie można ustawić stylu matplotlib")

            # Konfiguracja fontów
            plt.rcParams.update({
                'font.size': 10,
                'font.family': 'sans-serif',
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14
            })

            print("DEBUG: ✅ Matplotlib skonfigurowany")

        except Exception as e:
            print(f"WARNING: Błąd konfiguracji matplotlib: {e}")

    def test_matplotlib_compatibility(self):
        """Testuje kompatybilność matplotlib z tkinter"""
        try:
            print("DEBUG: Testuję kompatybilność matplotlib...")

            # Test 1: Backend
            current_backend = plt.get_backend()
            print(f"DEBUG: Aktualny backend: {current_backend}")

            # Test 2: Tworzenie Figure
            fig = Figure(figsize=(4, 3))
            print(f"DEBUG: ✅ Figure utworzona: {type(fig)}")

            # Test 3: Dodawanie subplot
            ax = fig.add_subplot(111)
            print(f"DEBUG: ✅ Subplot dodany: {type(ax)}")

            # Test 4: Prosty plot
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title('Test')
            print("DEBUG: ✅ Prosty wykres utworzony")

            # Test 5: tight_layout
            fig.tight_layout()
            print("DEBUG: ✅ tight_layout zadziałał")

            print("DEBUG: ✅ Wszystkie testy matplotlib przeszły")
            return True

        except Exception as e:
            print(f"ERROR: Test matplotlib nie powiódł się: {e}")
            import traceback
            print(f"TRACEBACK: {traceback.format_exc()}")
            return False