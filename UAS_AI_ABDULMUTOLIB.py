{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b03fc918-f6ac-441a-9a99-fa6998013f0c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Matplotlib is building the font cache; this may take a moment.\n"
     ]
    }
   ],
   "source": [
    "#import library dan package yang dibutuhkan\n",
    "\n",
    "import numpy as np #untuk perhitungan saintifik\n",
    "import matplotlib.pyplot as plt #untuk plotting\n",
    "from sklearn.linear_model import LinearRegression #import library LinearRegression dari scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "fc5ca914-fee0-4e68-87b3-b4de1095b50c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#buat data\n",
    "\n",
    "penjualan = np.array([6,5,5,4,4,3,2,2,2,1])\n",
    "harga = np.array([16000, 18000, 27000, 34000, 50000, 68000, 65000, 81000, 85000, 90000])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ed86ac46-8a8b-47de-a774-ae127942b925",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x27236861100>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYQAAAD4CAYAAADsKpHdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAZJElEQVR4nO3db2xVd37n8fenhsk4mRInxEHEZheqILb5ownJFaKLFO0O05ppR4M3SiRXmgZVaJEi1M1sV1S4T1Z9UIWIVdPNgyChsI2TTiehDCFoZhMGQf8rhb3EGTnAWLhlJtimwd3EGabjTYF+98H93s61Y+x7wfbNtT8v6eqc+73nd/w7kcLnnt/vnHsUEZiZmf1cvTtgZmafDQ4EMzMDHAhmZpYcCGZmBjgQzMwsLap3B27UXXfdFStXrqx3N8zMGsqpU6f+MSJaJ/usYQNh5cqVFIvFenfDzKyhSPrR9T7zkJGZmQEOBDMzSw4EMzMDHAhmZpYcCGZmBlQZCJKelvSepNOSvpG1OyUdlXQul3dUbN8taUBSv6SOivojkvrys+clKeu3SHot6yckrZzZwyw51DvEhl3HWbXzu2zYdZxDvUOz8WfMzBrStIEg6QHgPwPrgC8CX5W0GtgJHIuI1cCxfI+k+4Au4H5gE/CCpKbc3R5gG7A6X5uyvhX4KCLuBZ4Dnp2Ro6twqHeI7oN9DI2OEcDQ6BjdB/scCmZmqZozhF8E/jYifhoRV4G/AP4TsBnoyW16gM5c3wy8GhGfRMR5YABYJ2k5sCQi3o7Sb26/PKFNeV8HgI3ls4eZsvtIP2NXro2rjV25xu4j/TP5Z8zMGlY1gfAe8KikpZJuBX4VWAEsi4iLALm8O7dvAy5UtB/MWluuT6yPa5Oh8zGwdGJHJG2TVJRUHBkZqe4I0/DoWE11M7OFZtpAiIizlIZwjgJvAd8Hrk7RZLJv9jFFfao2E/uyNyIKEVFobZ30zuvruqeluaa6mdlCU9WkckTsi4iHI+JR4EPgHPBBDgORy0u5+SClM4iydmA46+2T1Me1kbQIuD3/zozZ0bGG5sVN42rNi5vY0bFmJv+MmVnDqvYqo7tz+W+Ax4BvAYeBLbnJFuCNXD8MdOWVQ6soTR6fzGGly5LW5/zAkxPalPf1OHA8ZvjZnp1r23jmsQdpa2lGQFtLM8889iCda9umbWtmthBU++N235a0FLgCbI+IjyTtAvZL2gq8DzwBEBGnJe0HzlAaWtoeEeXZ3KeAl4Bm4M18AewDXpE0QOnMoOumj2wSnWvbHABmZtehGf4iPmcKhUL4107NzGoj6VREFCb7zHcqm5kZ4EAwM7PkQDAzM8CBYGZmyYFgZmZAAz9T2apzqHeI3Uf6GR4d456WZnZ0rPGlt2Y2KQfCPFb+hdfyj/qVf+EVcCiY2ad4yGge8y+8mlktHAjzmH/h1cxq4UCYx/wLr2ZWCwfCPOZfeDWzWnhSeR4rTxz7KiMzq4YDYZ7zL7yaWbU8ZGRmZoADwczMkoeM5jnfqWxm1XIgzGO+U9nMauEho3nMdyqbWS2qCgRJ/1XSaUnvSfqWpM9LulPSUUnncnlHxfbdkgYk9UvqqKg/IqkvP3tekrJ+i6TXsn5C0soZP9IFyHcqm1ktpg0ESW3AfwEKEfEA0AR0ATuBYxGxGjiW75F0X35+P7AJeEFS+e6oPcA2YHW+NmV9K/BRRNwLPAc8OyNHt8D5TmUzq0W1Q0aLgGZJi4BbgWFgM9CTn/cAnbm+GXg1Ij6JiPPAALBO0nJgSUS8HREBvDyhTXlfB4CN5bMHu3G+U9nMajFtIETEEPA/gPeBi8DHEfE9YFlEXMxtLgJ3Z5M24ELFLgaz1pbrE+vj2kTEVeBjYOnEvkjaJqkoqTgyMlLtMS5YnWvbeOaxB2lraUZAW0szzzz2oCeUzWxS015llHMDm4FVwCjwp5K+PlWTSWoxRX2qNuMLEXuBvQCFQuFTn9un+U5lM6tWNUNGXwbOR8RIRFwBDgL/Hvggh4HI5aXcfhBYUdG+ndIQ02CuT6yPa5PDUrcDH97IAZmZ2Y2pJhDeB9ZLujXH9TcCZ4HDwJbcZgvwRq4fBrryyqFVlCaPT+aw0mVJ63M/T05oU97X48DxnGcwM7M5Mu2QUUSckHQAeAe4CvRSGrb5ArBf0lZKofFEbn9a0n7gTG6/PSLKF8M/BbwENANv5gtgH/CKpAFKZwZdM3J0ZmZWNTXqF/FCoRDFYrHe3TAzayiSTkVEYbLPfKeymZkBDgQzM0sOBDMzAxwIZmaWHAhmZgY4EMzMLDkQzMwMcCCYmVlyIJiZGeBAMDOz5EAwMzPAgWBmZsmBYGZmgAPBzMySA8HMzAAHgpmZJQeCmZkBDgQzM0vTBoKkNZLerXj9WNI3JN0p6aikc7m8o6JNt6QBSf2SOirqj0jqy8+el6Ss3yLptayfkLRyVo7WzMyua9pAiIj+iHgoIh4CHgF+CrwO7ASORcRq4Fi+R9J9QBdwP7AJeEFSU+5uD7ANWJ2vTVnfCnwUEfcCzwHPzsjRmZlZ1WodMtoI/F1E/AjYDPRkvQfozPXNwKsR8UlEnAcGgHWSlgNLIuLtiAjg5Qltyvs6AGwsnz2YmdncqDUQuoBv5fqyiLgIkMu7s94GXKhoM5i1tlyfWB/XJiKuAh8DSyf+cUnbJBUlFUdGRmrsupmZTaXqQJD0OeBrwJ9Ot+kktZiiPlWb8YWIvRFRiIhCa2vrNN0wM7NaLKph268A70TEB/n+A0nLI+JiDgddyvogsKKiXTswnPX2SeqVbQYlLQJuBz6s6UjM0qHeIXYf6Wd4dIx7WprZ0bGGzrVt0zc0W+BqGTL6dX42XARwGNiS61uANyrqXXnl0CpKk8cnc1jpsqT1OT/w5IQ25X09DhzPeQazmhzqHaL7YB9Do2MEMDQ6RvfBPg71DtW7a2afeVUFgqRbgV8GDlaUdwG/LOlcfrYLICJOA/uBM8BbwPaIuJZtngJepDTR/HfAm1nfByyVNAD8NnnFklmtdh/pZ+zKtXG1sSvX2H2kv049MmscVQ0ZRcRPmTDJGxH/l9JVR5Nt//vA709SLwIPTFL/f8AT1fTFbCrDo2M11c3sZ3ynss0r97Q011Q3s59xINi8sqNjDc2Lm8bVmhc3saNjTZ16ZNY4arnKyOwzr3w1ka8yMqudA8Hmnc61bQ4AsxvgQJjnfE2+mVXLgTCPla/JL1+GWb4mH3AomNmneFJ5HvM1+WZWCwfCPOZr8s2sFg6EeczX5JtZLRwI85ivyTezWnhSeR7zNflmVgsHwjzna/LNrFoeMjIzM8CBYGZmyYFgZmaAA8HMzJIDwczMgOofodki6YCkH0g6K+mXJN0p6aikc7m8o2L7bkkDkvoldVTUH5HUl589n89WJp+//FrWT0haOeNHamZmU6r2DOF/Am9FxL8DvgicpfTc42MRsRo4lu+RdB/QBdwPbAJekFS+O2oPsA1Yna9NWd8KfBQR9wLPAc/e5HGZmVmNpg0ESUuAR4F9ABHxzxExCmwGenKzHqAz1zcDr0bEJxFxHhgA1klaDiyJiLcjIoCXJ7Qp7+sAsLF89mBmZnOjmjOEXwBGgD+S1CvpRUm3Acsi4iJALu/O7duACxXtB7PWlusT6+PaRMRV4GNg6cSOSNomqSipODIyUuUhmplZNaoJhEXAw8CeiFgL/BM5PHQdk32zjynqU7UZX4jYGxGFiCi0trZO3WszM6tJNYEwCAxGxIl8f4BSQHyQw0Dk8lLF9isq2rcDw1lvn6Q+ro2kRcDtwIe1HoyZmd24aQMhIv4BuCCp/BOZG4EzwGFgS9a2AG/k+mGgK68cWkVp8vhkDitdlrQ+5weenNCmvK/HgeM5z2BmZnOk2h+3+y3gm5I+B/w98JuUwmS/pK3A+8ATABFxWtJ+SqFxFdgeEeXHdj0FvAQ0A2/mC0oT1q9IGqB0ZtB1k8dlZmY1UqN+ES8UClEsFuvdDTOzhiLpVEQUJvvMdyqbmRngQDAzs+RAMDMzwIFgZmbJgWBmZoADwczMkgPBzMwAB4KZmSUHgpmZAQ4EMzNLDgQzMwMcCGZmlhwIZmYGOBDMzCw5EMzMDHAgmJlZciCYmRngQDAzs1RVIEj6oaQ+Se9KKmbtTklHJZ3L5R0V23dLGpDUL6mjov5I7mdA0vOSlPVbJL2W9ROSVs7wcZrNa4d6h9iw6zirdn6XDbuOc6h3qN5dsgZUyxnCf4yIhyqexbkTOBYRq4Fj+R5J9wFdwP3AJuAFSU3ZZg+wDVidr01Z3wp8FBH3As8Bz974IZktLId6h+g+2MfQ6BgBDI2O0X2wz6FgNbuZIaPNQE+u9wCdFfVXI+KTiDgPDADrJC0HlkTE2xERwMsT2pT3dQDYWD57MLOp7T7Sz9iVa+NqY1eusftIf516ZI2q2kAI4HuSTknalrVlEXERIJd3Z70NuFDRdjBrbbk+sT6uTURcBT4Glk7shKRtkoqSiiMjI1V23Wx+Gx4dq6ludj3VBsKGiHgY+AqwXdKjU2w72Tf7mKI+VZvxhYi9EVGIiEJra+t0fTZbEO5paa6pbnY9VQVCRAzn8hLwOrAO+CCHgcjlpdx8EFhR0bwdGM56+yT1cW0kLQJuBz6s/XDMFp4dHWtoXtw0rta8uIkdHWvq1CNrVNMGgqTbJP18eR34FeA94DCwJTfbAryR64eBrrxyaBWlyeOTOax0WdL6nB94ckKb8r4eB47nPIOZTaNzbRvPPPYgbS3NCGhraeaZxx6kc23btG3NKi2qYptlwOs5x7sI+JOIeEvS/wH2S9oKvA88ARARpyXtB84AV4HtEVGe8XoKeAloBt7MF8A+4BVJA5TODLpm4NjMFozOtW0OALtpatQv4oVCIYrFYr27YWbWUCSdqrh9YBzfqWxmZoADwczMkgPBzMwAB4KZmSUHgpmZAQ4EMzNLDgQzMwMcCGZmlhwIZmYGOBDMzCw5EMzMDHAgmJlZciCYmRngQDAzs+RAMDMzwIFgZmbJgWBmZoADwczMUtWBIKlJUq+k7+T7OyUdlXQul3dUbNstaUBSv6SOivojkvrys+eVD2qWdIuk17J+QtLKGTxGMzOrQi1nCE8DZyve7wSORcRq4Fi+R9J9QBdwP7AJeEFSU7bZA2wDVudrU9a3Ah9FxL3Ac8CzN3Q0ZmZ2w6oKBEntwK8BL1aUNwM9ud4DdFbUX42ITyLiPDAArJO0HFgSEW9HRAAvT2hT3tcBYGP57MHMzOZGtWcIfwj8DvAvFbVlEXERIJd3Z70NuFCx3WDW2nJ9Yn1cm4i4CnwMLJ3YCUnbJBUlFUdGRqrsupmZVWPaQJD0VeBSRJyqcp+TfbOPKepTtRlfiNgbEYWIKLS2tlbZHTMzq8aiKrbZAHxN0q8CnweWSPpj4ANJyyPiYg4HXcrtB4EVFe3bgeGst09Sr2wzKGkRcDvw4Q0ek5mZ3YBpzxAiojsi2iNiJaXJ4uMR8XXgMLAlN9sCvJHrh4GuvHJoFaXJ45M5rHRZ0vqcH3hyQpvyvh7Pv/GpMwQzM5s91ZwhXM8uYL+krcD7wBMAEXFa0n7gDHAV2B4R17LNU8BLQDPwZr4A9gGvSBqgdGbQdRP9MjOzG6BG/SJeKBSiWCzWuxtmZg1F0qmIKEz2me9UNjMzwIFgZmbJgWBmZoADwczMkgPBzMwAB4KZmSUHgpmZATd3Y5qZfUYc6h1i95F+hkfHuKelmR0da+hc2zZ9Q7MKDgSzBneod4jug32MXSn9IMDQ6BjdB/sAHApWEw8ZmTW43Uf6/zUMysauXGP3kf469cgalQPBrMENj47VVDe7HgeCWYO7p6W5prrZ9TgQzBrcjo41NC9uGldrXtzEjo41deqRNSpPKps1uPLEsa8yspvlQDCbBzrXtjkA7KZ5yMjMzAAHgpmZpWkDQdLnJZ2U9H1JpyX9XtbvlHRU0rlc3lHRplvSgKR+SR0V9Uck9eVnz+ezlcnnL7+W9ROSVs7CsZqZ2RSqOUP4BPhSRHwReAjYJGk9sBM4FhGrgWP5Hkn3UXom8v3AJuAFSeVLIPYA24DV+dqU9a3ARxFxL/Ac8OzNH5qZmdVi2kCIkp/k28X5CmAz0JP1HqAz1zcDr0bEJxFxHhgA1klaDiyJiLej9CDnlye0Ke/rALCxfPZgZmZzo6o5BElNkt4FLgFHI+IEsCwiLgLk8u7cvA24UNF8MGttuT6xPq5NRFwFPgaW3sDxmJnZDaoqECLiWkQ8BLRT+rb/wBSbT/bNPqaoT9Vm/I6lbZKKkoojIyPT9NrMzGpR01VGETEK/Dmlsf8PchiIXF7KzQaBFRXN2oHhrLdPUh/XRtIi4Hbgw0n+/t6IKEREobW1tZaum5nZNKq5yqhVUkuuNwNfBn4AHAa25GZbgDdy/TDQlVcOraI0eXwyh5UuS1qf8wNPTmhT3tfjwPGcZzAzszlSzZ3Ky4GevFLo54D9EfEdSW8D+yVtBd4HngCIiNOS9gNngKvA9ogo/zbvU8BLQDPwZr4A9gGvSBqgdGbQNRMHZ2Zm1VOjfhEvFApRLBbr3Q0zs4Yi6VREFCb7zL9lZGYNyY8NnXkOBDNrOH5s6OzwbxmZWcPxY0NnhwPBzBqOHxs6OxwIZtZw/NjQ2eFAMLOG48eGzg5PKptZw/FjQ2eHA8HMGpIfGzrzPGRkZmaAA8HMzJIDwczMAAeCmZklB4KZmQEOBDMzSw4EMzMDHAhmZpYcCGZmBjgQzMwsTRsIklZI+jNJZyWdlvR01u+UdFTSuVzeUdGmW9KApH5JHRX1RyT15WfPS1LWb5H0WtZPSFo5C8dqZmZTqOYM4Srw3yLiF4H1wHZJ9wE7gWMRsRo4lu/Jz7qA+4FNwAuSyj9LuAfYBqzO16asbwU+ioh7geeAZ2fg2MzMrAbTBkJEXIyId3L9MnAWaAM2Az25WQ/QmeubgVcj4pOIOA8MAOskLQeWRMTbERHAyxPalPd1ANhYPnswM7O5UdMcQg7lrAVOAMsi4iKUQgO4OzdrAy5UNBvMWluuT6yPaxMRV4GPgaWT/P1tkoqSiiMjI7V03czMplF1IEj6AvBt4BsR8eOpNp2kFlPUp2ozvhCxNyIKEVFobW2drstmZlaDqgJB0mJKYfDNiDiY5Q9yGIhcXsr6ILCionk7MJz19knq49pIWgTcDnxY68GYmdmNq+YqIwH7gLMR8QcVHx0GtuT6FuCNinpXXjm0itLk8ckcVrosaX3u88kJbcr7ehw4nvMMZmaTOtQ7xIZdx1m187ts2HWcQ71D9e5Sw6vmiWkbgN8A+iS9m7XfBXYB+yVtBd4HngCIiNOS9gNnKF2htD0irmW7p4CXgGbgzXxBKXBekTRA6cyg6+YOy8zms0O9Q3Qf7GPsSumflqHRMboP9gH4KWo3QY36RbxQKESxWKx3N8ysDjbsOs7Q6Nin6m0tzfzNzi/VoUeNQ9KpiChM9pnvVDazhjM8SRhMVbfqOBDMrOHc09JcU92q40Aws4azo2MNzYubxtWaFzexo2NNnXo0N2Z7Ir2aSWUzs8+U8sTx7iP9DI+OcU9LMzs61szrCeW5mEh3IJhZQ+pc2zavA2Ci3Uf6/zUMysauXGP3kf4Z++/gISMzswYwFxPpDgQzswYwFxPpDgQzswYwFxPpnkMwM2sAczGR7kAwM2sQsz2R7iEjMzMDHAhmZpYcCGZmBjgQzMwsORDMzAxo4OchSBoBfnSDze8C/nEGu9MIfMwLg495YbiZY/63ETHpQ+kbNhBuhqTi9R4QMV/5mBcGH/PCMFvH7CEjMzMDHAhmZpYWaiDsrXcH6sDHvDD4mBeGWTnmBTmHYGZmn7ZQzxDMzGwCB4KZmQELLBAk/S9JlyS9V+++zBVJKyT9maSzkk5LerrefZptkj4v6aSk7+cx/169+zQXJDVJ6pX0nXr3ZS5I+qGkPknvSirWuz9zQVKLpAOSfpD/T//SjO5/Ic0hSHoU+AnwckQ8UO/+zAVJy4HlEfGOpJ8HTgGdEXGmzl2bNZIE3BYRP5G0GPhr4OmI+Ns6d21WSfptoAAsiYiv1rs/s03SD4FCRCyYm9Ik9QB/FREvSvoccGtEjM7U/hfUGUJE/CXwYb37MZci4mJEvJPrl4GzwLx+MnmU/CTfLs7XvP7mI6kd+DXgxXr3xWaHpCXAo8A+gIj455kMA1hggbDQSVoJrAVO1Lkrsy6HT94FLgFHI2K+H/MfAr8D/Eud+zGXAviepFOSttW7M3PgF4AR4I9yaPBFSbfN5B9wICwQkr4AfBv4RkT8uN79mW0RcS0iHgLagXWS5u0QoaSvApci4lS9+zLHNkTEw8BXgO05JDyfLQIeBvZExFrgn4CdM/kHHAgLQI6jfxv4ZkQcrHd/5lKeUv85sKm+PZlVG4Cv5Zj6q8CXJP1xfbs0+yJiOJeXgNeBdfXt0awbBAYrznYPUAqIGeNAmOdygnUfcDYi/qDe/ZkLklolteR6M/Bl4Ad17dQsiojuiGiPiJVAF3A8Ir5e527NKkm35UUS5LDJrwDz+urBiPgH4IKkNVnaCMzoxSGLZnJnn3WSvgX8B+AuSYPAf4+IffXt1azbAPwG0Jdj6gC/GxH/u35dmnXLgR5JTZS+9OyPiAVxKeYCsgx4vfR9h0XAn0TEW/Xt0pz4LeCbeYXR3wO/OZM7X1CXnZqZ2fV5yMjMzAAHgpmZJQeCmZkBDgQzM0sOBDMzAxwIZmaWHAhmZgbA/wfPdX3KJxAwggAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#buat plot\n",
    "%matplotlib inline\n",
    "plt.scatter (penjualan, harga)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "45fbb141-9fa3-426e-b751-721cafe3008f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#buat permodelan regresi\n",
    "\n",
    "penjualan = penjualan.reshape (-1,1) #kita tukar baris dan kolom variabel ini, agar bisa dikalikan dalam operasi matriks\n",
    "#untuk lebih lengkapnya baca teori soal perhitungan regresi linier\n",
    "\n",
    "linreg = LinearRegression()\n",
    "linreg.fit(penjualan, harga)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "de2c3bee-eafd-45c1-a034-a15a0ea56df2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x27236fcfca0>]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plot hasil regresi\n",
    "\n",
    "plt.scatter(penjualan, harga, color='red')\n",
    "plt.plot(penjualan, linreg.predict(penjualan))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "38c5e5ce-7a62-495c-8532-7317914d77f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import library dan package yang dibutuhkan\n",
    "\n",
    "import pandas as pd #untuk dataframe\n",
    "import pylab as pl #untuk plotting\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f990c4df-6b92-4b7e-a874-a2501aca3f1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'wget' is not recognized as an internal or external command,\n",
      "operable program or batch file.\n"
     ]
    }
   ],
   "source": [
    "#download data\n",
    "\n",
    "!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d28a11e3-e975-41a2-ab8e-a2d5c3daa44f",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'FuelConsumption.csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-b8df829c61a7>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mdf\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"FuelConsumption.csv\"\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;31m#membaca data\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;31m# melihat 5 baris pertama data\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36mread_csv\u001b[1;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)\u001b[0m\n\u001b[0;32m    608\u001b[0m     \u001b[0mkwds\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkwds_defaults\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    609\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 610\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    611\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    612\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_read\u001b[1;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[0;32m    460\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    461\u001b[0m     \u001b[1;31m# Create the parser.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 462\u001b[1;33m     \u001b[0mparser\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    463\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    464\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[0;32m    817\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    818\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 819\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    820\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    821\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[1;34m(self, engine)\u001b[0m\n\u001b[0;32m   1048\u001b[0m             )\n\u001b[0;32m   1049\u001b[0m         \u001b[1;31m# error: Too many arguments for \"ParserBase\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1050\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mmapping\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# type: ignore[call-arg]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1051\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1052\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_failover_to_python\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, src, **kwds)\u001b[0m\n\u001b[0;32m   1865\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1866\u001b[0m         \u001b[1;31m# open handles\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1867\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_open_handles\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1868\u001b[0m         \u001b[1;32massert\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhandles\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1869\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;32min\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;34m\"storage_options\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"encoding\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"memory_map\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"compression\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_open_handles\u001b[1;34m(self, src, kwds)\u001b[0m\n\u001b[0;32m   1360\u001b[0m         \u001b[0mLet\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mreaders\u001b[0m \u001b[0mopen\u001b[0m \u001b[0mIOHanldes\u001b[0m \u001b[0mafter\u001b[0m \u001b[0mthey\u001b[0m \u001b[0mare\u001b[0m \u001b[0mdone\u001b[0m \u001b[1;32mwith\u001b[0m \u001b[0mtheir\u001b[0m \u001b[0mpotential\u001b[0m \u001b[0mraises\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1361\u001b[0m         \"\"\"\n\u001b[1;32m-> 1362\u001b[1;33m         self.handles = get_handle(\n\u001b[0m\u001b[0;32m   1363\u001b[0m             \u001b[0msrc\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1364\u001b[0m             \u001b[1;34m\"r\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\common.py\u001b[0m in \u001b[0;36mget_handle\u001b[1;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[0;32m    640\u001b[0m                 \u001b[0merrors\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"replace\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    641\u001b[0m             \u001b[1;31m# Encoding\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 642\u001b[1;33m             handle = open(\n\u001b[0m\u001b[0;32m    643\u001b[0m                 \u001b[0mhandle\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    644\u001b[0m                 \u001b[0mioargs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'FuelConsumption.csv'"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(\"FuelConsumption.csv\") #membaca data\n",
    "\n",
    "# melihat 5 baris pertama data\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1360c474-dc31-4ec8-94a7-86f67627b497",
   "metadata": {},
   "outputs": [],
   "source": [
    "#kita ambil kolom mana saja yang akan kita analisis, dan membuang sisanya\n",
    "\n",
    "cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','CO2EMISSIONS']]\n",
    "cdf.head(9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a18ba175-7fa6-485c-b864-55474a29b622",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Kita plot hubungannya\n",
    "\n",
    "plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"FUELCONSUMPTION_CITY\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2559a868-bd12-4898-82e3-c24ad0933437",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Kita plot hubungannya\n",
    "\n",
    "plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cdea8d22-0374-4ac0-9d2a-bd88dcc6f2b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Membagi data\n",
    "\n",
    "msk = np.random.rand(len(df)) < 0.8\n",
    "train = cdf[msk]\n",
    "test = cdf[~msk]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c8a1022-b07d-4e6b-b192-5c3b61a4bb24",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1590755a-0c87-4bd1-b8f7-dc5cbef52031",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Membuat model regresi\n",
    "regr = LinearRegression()\n",
    "train_x = np.asanyarray(train[['ENGINESIZE']])\n",
    "train_y = np.asanyarray(train[['CO2EMISSIONS']])\n",
    "regr.fit (train_x, train_y)\n",
    "\n",
    "# Koefisien model\n",
    "print ('Coefficients: ', regr.coef_)\n",
    "print ('Intercept: ',regr.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98028c69-c157-445a-a9ef-05fb75882a52",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Plot hasil regresi\n",
    "\n",
    "plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')\n",
    "plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a26c2cc0-46ca-443c-bbd7-93413f4408c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Menghitung error\n",
    "\n",
    "from sklearn.metrics import r2_score\n",
    "\n",
    "test_x = np.asanyarray(test[['ENGINESIZE']])\n",
    "test_y = np.asanyarray(test[['CO2EMISSIONS']])\n",
    "test_y_ = regr.predict(test_x)\n",
    "\n",
    "print(\"Mean absolute error: %.2f\" % np.mean(np.absolute(test_y_ - test_y)))\n",
    "print(\"Residual sum of squares (MSE): %.2f\" % np.mean((test_y_ - test_y) ** 2))\n",
    "print(\"R2-score: %.2f\" % r2_score(test_y_ , test_y) )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed698abf-efc4-48dc-89b0-fe05d385ae3a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
