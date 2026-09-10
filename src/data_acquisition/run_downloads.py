import subprocess
import sys

def run_script(script_name):
    print(f"\n=== Lancement de {script_name} ===\n")
    result = subprocess.run([sys.executable, script_name], check=True)
    print(f"\n=== {script_name} terminé avec succès ! ===\n")

def main():
    try:
        run_script("download_sentinel-1.py")
        run_script("download_sentinel-2.py")
        print("\n Tous les téléchargements sont terminés sans erreur")
    except subprocess.CalledProcessError as e:
        print("\n Erreur pendant l'exécution :", e)

if __name__ == "__main__":
    main()
