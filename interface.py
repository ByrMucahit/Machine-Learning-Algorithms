from tkinter import *
from tkinter import ttk

root = Tk()
root.geometry("1352x650+0+0")
root.title("Vehicle Trading Management System")
root.configure(background="black")


Tops = Frame(root, width=1350, height=100, bd=4, relief="raise")
Tops.pack(side=TOP)

lblInfo = Label(Tops, font=('Comic Sans MS',50),text="Vehicle Sales Trading Management System",bd=5,anchor='w')
lblInfo.grid(row=0,column=0)


bottom = Frame(root, width=1352, height=550, bd=1, relief="raise")
bottom.pack(side=TOP)

#Root is divided into two bottomleft and bottom right
#bottomleft is divided into bottomleftTop and bottomleftBottom
#BottomleftTop is further divided into bottomLeftTopL and bottomLeftTopR
#BottomleftBottom is further divided into bottomLeftBottomL and bottomLeftBottomR
bottomLeft = Frame(bottom, width=1000, height=650,bd=1, relief="raise")
bottomLeft.pack(side=LEFT)
#========================================================================================================================#
bottomLeftTop = Frame(bottomLeft, width=1000, height=300, relief="raise")
bottomLeftTop.pack(side=TOP)

bottomLeftTopL = Frame(bottomLeftTop, width=500, height=400, bd=1, relief="raise")
bottomLeftTopL.pack(side=LEFT)

bottomLeftTopR = Frame(bottomLeftTop, width=500, height=400, bd=1, relief="raise")
bottomLeftTopR.pack(side=RIGHT)
#========================================================================================================================#

bottomLeftBottom = Frame(bottomLeft, width=1000, height=300, bd=2, relief="raise")
bottomLeftBottom.pack(side=BOTTOM)

bottomLeftBottomL = Frame(bottomLeftBottom, width=500, height=300, bd=2, relief="raise")
bottomLeftBottomL.pack(side=LEFT)

bottomLeftBottomR = Frame(bottomLeftBottom, width=500, height=300, bd=2, relief="raise")
bottomLeftBottomR.pack(side=RIGHT)

#========================================================================================================================#

bottomRight = Frame(bottom, width=352, height=650, bd=1, relief="raise")
bottomRight.pack(side=RIGHT)

#=====================================Exit Fn.===========================================================================#
def iExit():
    iExit=messagebox.askyesno("Vehicle Trading System","Confirm if you want to exit")
    if iExit > 0:
        root.destroy()
        return
    
#=====================================Reset Fn.==========================================================================#
def Reset():
    Modified.set('0')
    Stereo.set('0')
    Customised.set('0')
    Leather.set('0')
    GPS.set('0')
    CostofCar.set("0")
    CarMileage.set("0")
    CustomerName.set("")
    CustomerAddress.set("")
    CustomerPostcode.set("")
    CustomerTelephone.set("")
    VAT.set("")
    Discount.set("")
    Tax.set("")
    SubTotal.set("")
    Total.set("")
    txtReceipt.delete("1.0",END)

    var1.set(0)
    var2.set(0)
    var3.set(0)
    var4.set(0)
    var5.set(0)
    var6.set(0)
    var7.set(0)
    var8.set(0)
    var9.set(0)

#=============================================CarCost====================================================================#
def CarCost():
    if (var7.get() == 'Swift'):
        myCar = float(525000)
        CostofCar.set(myCar)
    elif (var7.get() == 'City'):
        myCar = float(835000)
        CostofCar.set(myCar)
    elif (var7.get() == 'Ecosport'):
        myCar = float(786000)
        CostofCar.set(myCar)
    elif (var7.get() == 'Safari'):
        myCar = float(425000)
        CostofCar.set(myCar)
    elif (var7.get() == 'Micra'):
        myCar = float(230000)
        CostofCar.set(myCar)
    elif (var7.get() == 'A4'):
        myCar = float(2700000)
        CostofCar.set(myCar)
    elif (var7.get() == 'M5'):
        myCar = float(4400000)
        CostofCar.set(myCar)
#------------------------------Swift----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'Swift'):
        myCar = float(525000)
       # iMile = 'Maruti Suzuki'
        CostofCar.set(myCar)
        CarMileage.set('Maruti Suzuki')

    if (var18.get() == '5001-20000' and var7.get() == 'Swift'):
        myCar = float(525000)
        #iMile = 'Maruti Suzuki'
        CostofCar.set(myCar)
        CarMileage.set('Maruti Suzuki')

    if (var18.get() == '20000-100000' and var7.get() == 'Swift'):
        myCar = float(525000)
        #iMile = 'Maruti Suzuki'
        CostofCar.set(myCar)
        CarMileage.set('Maruti Suzuki')

    if (var18.get() == '1000001 or more' and var7.get() == 'Swift'):
        myCar = float(525000)
        #iMile = 'Maruti Suzuki'
        CostofCar.set(myCar)
        CarMileage.set('Maruti Suzuki')

#------------------------------City----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'City'):
        myCar = float(835000)
        iMile = 'Honda'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '5001-20000' and var7.get() == 'City'):
        myCar = float(810000)
        iMile = 'Honda'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '20000-100000' and var7.get() == 'City'):
        myCar = float(740000)
        iMile = 'Honda'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '1000001 or more' and var7.get() == 'City'):
        myCar = float(650000)
        iMile = 'Honda'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

#------------------------------Ecosport----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'Ecosport'):
        myCar = float(835000)
        iMile = 'Ford'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '5001-20000' and var7.get() == 'Ecosport'):
        myCar = float(810000)
        iMile = 'Ford'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '20000-100000' and var7.get() == 'Ecosport'):
        myCar = float(740000)
        iMile = 'Ford'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '1000001 or more' and var7.get() == 'Ecosport'):
        myCar = float(650000)
        iMile = 'Ford'
        CostofCar.set(myCar)
        CarMileage.set(iMile)


#------------------------------Safari----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'Safari'):
        myCar = float(425000)
        iMile = 'Tata'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '5001-20000' and var7.get() == 'Safari'):
        myCar = float(400000)
        iMile = 'Tata'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '20000-100000' and var7.get() == 'Safari'):
        myCar = float(380000)
        iMile = 'Tata'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '1000001 or more' and var7.get() == 'Safari'):
        myCar = float(310000)
        iMile = 'Tata'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

#------------------------------Micra----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'Micra'):
        myCar = float(230000)
        iMile = 'Nissan'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '5001-20000' and var7.get() == 'Micra'):
        myCar = float(210000)
        iMile = 'Nissan'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '20000-100000' and var7.get() == 'Micra'):
        myCar = float(170000)
        iMile = 'Nissan'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '1000001 or more' and var7.get() == 'Micra'):
        myCar = float(140000)
        iMile = 'Nissan'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

#------------------------------A4----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'A4'):
        myCar = float(2700000)
        iMile = 'Nissan'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '5001-20000' and var7.get() == 'A4'):
        myCar = float(2690000)
        iMile = 'A4'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '20000-100000' and var7.get() == 'A4'):
        myCar = float(2600000)
        iMile = 'A4'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '1000001 or more' and var7.get() == 'A4'):
        myCar = float(2520000)
        iMile = 'A4'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

#------------------------------M5----------------------------------
    if (var18.get() == '100-5000' and var7.get() == 'M5'):
        myCar = float(4400000)
        iMile = 'BMW'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '5001-20000' and var7.get() == 'M5'):
        myCar = float(4390000)
        iMile = 'BMW'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '20000-100000' and var7.get() == 'M5'):
        myCar = float(4300000)
        iMile = 'BMW'
        CostofCar.set(myCar)
        CarMileage.set(iMile)

    if (var18.get() == '1000001 or more' and var7.get() == 'M5'):
        myCar = float(4320000)
        iMile = 'BMW'
        CostofCar.set(myCar)
        CarMileage.set(iMile)
#==========================================================================================================#
    if (var1.get() == 1):
        Modified.set(22000)
    elif (var1.get() == 0):
        Modified.set(0)
    if (var2.get() == 1):
        Stereo.set(8570)
    elif (var2.get() == 0):
        Stereo.set(0)
    if (var3.get() == 1):
        Leather.set(9500)
    elif (var3.get() == 0):
        Leather.set(0)
    if (var4.get() == 1):
        Customised.set(18755)
    elif (var4.get() == 0):
        Customised.set(0)
    if (var5.get() == 1):
        GPS.set(19550)
    elif (var5.get() == 0):
        GPS.set(0)
    if (var8.get() == 1):
        VAT.set("Yes")
    elif (var8.get() == 0):
        VAT.set("No")
    if (var9.get() == 1):
        Discount.set("Yes")
    elif (var9.get() == 0):
        Discount.set("No")

    Item1 = float(CostofCar.get())
    Item2 = (CarMileage.get())
    Item3 = float(Modified.get())
    Item4 = float(Stereo.get())
    Item5 = float(Leather.get())
    Item6 = float(Customised.get())
    Item7 = float(GPS.get())
    Item8 = "Rs." , str('%.2f'%((Item1) + Item3  + Item4  + Item5 + Item6 + Item7))
    Item9 = (((Item1) + Item3  + Item4  + Item5 + Item6 + Item7) * 0.45)/100
    Item9 = "Rs." , str('%.2f'%(Item9))
    Item10 = (((Item1) + Item3  + Item4  + Item5 + Item6 + Item7) * 0.45)/100
    Item11 = ((Item1) + Item3  + Item4  + Item5 + Item6 + Item7)
    Item12 = "Rs." , str('%.2f'%(Item10+Item11))
    SubTotal.set(Item8)
    Tax.set(Item9)
    Total.set(Item12)
    

#=====================================Receipt Fn.==============================================================#
def Receipt():
    txtReceipt.delete("1.0",END)
    txtReceipt.insert(END,'Items\t\t\t\t'+"Cost of Items \n\n")
    txtReceipt.insert(END,'========================================' "\n")
    txtReceipt.insert(END,'Customer Name: \t\t\t\t'+CustomerName.get()+"\n")
    txtReceipt.insert(END,'========================================'"\n")
    txtReceipt.insert(END,'Type of Car: \t\t\t\t'+var7.get()+"\n")
    txtReceipt.insert(END,'Cost of Car: \t\t\t\t'+CostofCar.get()+"\n")
    txtReceipt.insert(END,'Total Mileage: \t\t\t\t'+var18.get()+"\n")
    txtReceipt.insert(END,'Trade in value: \t\t\t\t'+CarMileage.get()+"\n")
    txtReceipt.insert(END,'Cost of Modified: \t\t\t\t'+Modified.get()+"\n")
    txtReceipt.insert(END,'Cost of Stereo: \t\t\t\t'+Stereo.get()+"\n")
    txtReceipt.insert(END,'Cost of Leather: \t\t\t\t'+Leather.get()+"\n")
    txtReceipt.insert(END,'Cost to Customised: \t\t\t\t'+Customised.get()+"\n")
    txtReceipt.insert(END,'Cost of GPS: \t\t\t\t'+GPS.get()+"\n")
    txtReceipt.insert(END,'========================================'"\n")
    txtReceipt.insert(END,'Tax: \t\t\t\t'+Tax.get()+"\n")
    txtReceipt.insert(END,'SubTotal: \t\t\t\t'+SubTotal.get()+"\n")
    txtReceipt.insert(END,'Total Cost: \t\t\t\t'+Total.get()+"\n")
    txtReceipt.insert(END,'========================================'"\n")
#=====================================Frame 1===================================================================#
#Variables
    
CustomerName = StringVar()
CustomerAddress = StringVar()
CustomerPostcode = StringVar()
CustomerTelephone = StringVar()
#-------------------------------
lblName = Label(bottomLeftTopL, font=('arial', 16,'bold'),text="Name:", fg="black", width= 15, bd=10, anchor='w')
lblName.grid(row=0, column=0)
txtName = Entry(bottomLeftTopL, font=('arial', 16,'bold'), bd=2, width = 24, bg="white", justify='left', textvariable=CustomerName)
txtName.grid(row=0, column=1)

lblAddress = Label(bottomLeftTopL, font=('arial', 16,'bold'),text="Address:", fg="black", width= 15, bd=10, anchor='w')
lblAddress.grid(row=1, column=0)
txtAddress = Entry(bottomLeftTopL, font=('arial', 16,'bold'), bd=2, width = 24, bg="white", justify='left', textvariable=CustomerAddress)
txtAddress.grid(row=1, column=1)

lblPostcode = Label(bottomLeftTopL, font=('arial', 16,'bold'),text="Pincode:", fg="black", width= 15, bd=10, anchor='w')
lblPostcode.grid(row=2, column=0)
txtPostcode = Entry(bottomLeftTopL, font=('arial', 16,'bold'), bd=2, width = 24, bg="white", justify='left', textvariable=CustomerPostcode)
txtPostcode.grid(row=2, column=1)

lblTelephone = Label(bottomLeftTopL, font=('arial', 16,'bold'),text="Mobile:", fg="black", width= 15, bd=10, anchor='w')
lblTelephone.grid(row=3, column=0)
txtTelephone = Entry(bottomLeftTopL, font=('arial', 16,'bold'), bd=2, width = 24, bg="white", justify='left', textvariable=CustomerTelephone)
txtTelephone.grid(row=3, column=1)
#==================================================Frame 3==================================================================================
#Variables
Modified = StringVar()
Stereo = StringVar()
Customised = StringVar()
Leather = StringVar()
GPS = StringVar()
#---------------------

Modified.set('0')
Stereo.set('0')
Customised.set('0')
Leather.set('0')
GPS.set('0')

var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var4 = IntVar()
var5 = IntVar()

lblModified = Checkbutton(bottomLeftBottomL, font=('arial', 16,'bold'),text="Modified", fg="black", width= 20, bd=10, anchor='w',
                          onvalue=1, offvalue=0, variable=var1)
lblModified.grid(row=0, column=0)
txtModified = Entry(bottomLeftBottomL, font=('arial', 16,'bold'), bd=2, width = 14, bg="white", justify='left', textvariable=Modified)
txtModified.grid(row=0, column=1)

lblStereo = Checkbutton(bottomLeftBottomL, font=('arial', 16,'bold'),text="Music System", fg="black", width= 20, bd=10, anchor='w',
                        onvalue=1, offvalue=0, variable=var2)
lblStereo.grid(row=1, column=0)
txtStereo = Entry(bottomLeftBottomL, font=('arial', 16,'bold'), bd=2, width = 14, bg="white", justify='left', textvariable=Stereo)
txtStereo.grid(row=1, column=1)

lblLeather = Checkbutton(bottomLeftBottomL, font=('arial', 16,'bold'),text="Leather Interior", fg="black", width= 20, bd=10, anchor='w',
                         onvalue=1, offvalue=0, variable=var3)
lblLeather.grid(row=2, column=0)
txtLeather = Entry(bottomLeftBottomL, font=('arial', 16,'bold'), bd=2, width = 14, bg="white", justify='left', textvariable=Leather)
txtLeather.grid(row=2, column=1)

lblCustomised = Checkbutton(bottomLeftBottomL, font=('arial', 16,'bold'),text="Customised", fg="black", width= 20, bd=10, anchor='w',
                            onvalue=1, offvalue=0, variable=var4)
lblCustomised.grid(row=3, column=0)
txtCustomised = Entry(bottomLeftBottomL, font=('arial', 16,'bold'), bd=2, width = 14, bg="white", justify='left', textvariable=Customised)
txtCustomised.grid(row=3, column=1)

lblGPS = Checkbutton(bottomLeftBottomL, font=('arial', 16,'bold'),text="GPS Maps", fg="black", width= 20, bd=10, anchor='w',
                     onvalue=1, offvalue=0, variable=var5)
lblGPS.grid(row=4, column=0)
txtGPS = Entry(bottomLeftBottomL, font=('arial', 16,'bold'), bd=2, width = 14, bg="white", justify='left', textvariable=GPS)
txtGPS.grid(row=4, column=1)

btnTotalCost=Button(bottomLeftBottomL,pady=8, bd=2, fg="black",font=('arial',16,'bold'),width=13,text="Total", bg="white",command=CarCost).grid(row=5, column=0)
btnTotalReceipt=Button(bottomLeftBottomL,pady=8, bd=2, fg="black",font=('arial',16,'bold'),width=13,text="Receipt", bg="white",command=Receipt).grid(row=5, column=1)


#==================================================Frame 2==================================================================================
var6 = IntVar()
var7 = StringVar()
var18 = StringVar()

CostofCar = StringVar()
CarMileage = StringVar()

CostofCar.set("0")
CarMileage.set("0")

lblChooseCar = Label(bottomLeftTopR, font=('arial', 16,'bold'),text="Choose a Car", fg="black", width= 13, bd=14, anchor='w')
lblChooseCar.grid(row=0, column=0)
choChooseCar = ttk.Combobox(bottomLeftTopR,textvariable = var7 ,state='readonly', font=('arial', 20,'bold'), width=12)
choChooseCar['value']=['','Swift','City','Ecosport','Safari','Micra','A4','M5']
choChooseCar.current(0)
choChooseCar.grid(row=1,column=0)

lblCostofCar = Label(bottomLeftTopR, font=('arial', 16,'bold'),text="Cost of Car", fg="black", width= 13, bd=14, anchor='w')
lblCostofCar.grid(row=2, column=0)
txtCostofCar = Entry(bottomLeftTopR, font=('arial', 16,'bold'), bd=2,width=16,bg="white", justify='left', textvariable=CostofCar)
txtCostofCar.grid(row=3,column=0)


lblTradeInaCar = Label(bottomLeftTopR, font=('arial', 16,'bold'),text="Preferred Mileage Range", fg="black", width= 13, bd=14, anchor='w')
lblTradeInaCar.grid(row=0, column=1)
choTradeInaCar = ttk.Combobox(bottomLeftTopR,textvariable = var18, state='readonly', font=('arial', 20,'bold'), width=12)
choTradeInaCar['value']=['','100-5000','5001-20000','20000-100000','100001 or more']
choTradeInaCar.current(0)
choTradeInaCar.grid(row=1,column=1)

lblCarMileage = Label(bottomLeftTopR, font=('arial', 16,'bold'),text="Brand Name", fg="black", width= 13, bd=14, anchor='w')
lblCarMileage.grid(row=2, column=1)
txtCarMileage = Entry(bottomLeftTopR, font=('arial', 16,'bold'), bd=2,width=16,bg="white", justify='left', textvariable=CarMileage)
txtCarMileage.grid(row=3,column=1)

#==================================================Frame 4==================================================================================

VAT = StringVar()
Discount = StringVar()
Tax = StringVar()
SubTotal = StringVar()
Total = StringVar()

var8 = IntVar()
var9 = IntVar()

lblVAT = Checkbutton(bottomLeftBottomR, font=('arial', 16,'bold'),text="VAT", fg="black", width= 13, bd=12, anchor='w',onvalue=1, offvalue=0, variable=var8)
lblVAT.grid(row=0, column=0)
txtVAT = Entry(bottomLeftBottomR, font=('arial', 16,'bold'), bd=2, width = 17, bg="white", justify='left', textvariable=VAT)
txtVAT.grid(row=0, column=1)

lblDiscount = Checkbutton(bottomLeftBottomR, font=('arial', 16,'bold'),text="Discount", fg="black", width= 13, bd=12, anchor='w',onvalue=1, offvalue=0, variable=var9)
lblDiscount.grid(row=1, column=0)
txtDiscount = Entry(bottomLeftBottomR, font=('arial', 16,'bold'), bd=2, width = 17, bg="white", justify='left', textvariable=Discount)
txtDiscount.grid(row=1, column=1)

lblTax = Label(bottomLeftBottomR, font=('arial', 16,'bold'),text="Tax", fg="black", width= 13, bd=12, anchor='w')
lblTax.grid(row=2, column=0)
txtTax = Entry(bottomLeftBottomR, font=('arial', 16,'bold'), bd=2, width = 17, bg="white", justify='left', textvariable=Tax)
txtTax.grid(row=2, column=1)

lblSubTotal = Label(bottomLeftBottomR, font=('arial', 16,'bold'),text="Sub Total", fg="black", width= 13, bd=12, anchor='w')
lblSubTotal.grid(row=3, column=0)
txtSubTotal = Entry(bottomLeftBottomR, font=('arial', 16,'bold'), bd=2, width = 17, bg="white", justify='left', textvariable=SubTotal)
txtSubTotal.grid(row=3, column=1)

lblTotal = Label(bottomLeftBottomR, font=('arial', 16,'bold'),text="Total", fg="black", width= 13, bd=12, anchor='w')
lblTotal.grid(row=4, column=0)
txtTotal = Entry(bottomLeftBottomR, font=('arial', 16,'bold'), bd=2, width = 17, bg="white", justify='left', textvariable=Total)
txtTotal.grid(row=4, column=1)

btnReset=Button(bottomLeftBottomR,pady=8, bd=2, fg="black",font=('arial',16,'bold'),width=13,text="Reset", bg="white",command=Reset).grid(row=6,column=0)
btnExit=Button(bottomLeftBottomR,pady=8, bd=2, fg="black",font=('arial',16,'bold'),width=13,text="Exit", bg="white",command=iExit).grid(row=6,column=1)


#==================================================Frame 5==================================================================================
lblReceipt = Label(bottomRight,font=('arial',16,'bold'),text="Receipt:",bd=2,anchor='w')
lblReceipt.grid(row=0,column=0, sticky=W)
txtReceipt = Text(bottomRight, font=('arial', 11,'bold'), bd=8, width = 46, height= 40, bg="white")
txtReceipt.grid(row=1, column=0)


root.mainloop()